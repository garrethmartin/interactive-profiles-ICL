import numpy as np
import holoviews as hv
import panel as pn
import matplotlib as mpl
import h5py
import scipy.stats
from concave_hull import concave_hull_indexes
from scipy.interpolate import splprep, splev

hv.extension('bokeh')
pn.extension()

# load from hdf5
with h5py.File('parameter_exploration_fine.hdf5', 'r') as f:
    results_fine = {}
    for label in f.keys():
        results_fine[label] = {}
        for vary in f[label].keys():
            grp_vary = f[label][vary]
            if vary == 'fiducial':
                results_fine[label][vary] = {key: grp_vary[key][()] for key in grp_vary.keys()}
            else:
                param_values = grp_vary['param_values'][()]
                medians_e = grp_vary['medians_e'][()]
                p16_e = grp_vary['p16_e'][()]
                p84_e = grp_vary['p84_e'][()]
                p3low_e = grp_vary['p3low_e'][()]
                p3high_e = grp_vary['p3high_e'][()]
                medians_h = grp_vary['medians_h'][()]
                p16_h = grp_vary['p16_h'][()]
                p84_h = grp_vary['p84_h'][()]
                p3low_h = grp_vary['p3low_h'][()]
                p3high_h = grp_vary['p3high_h'][()]
                percs_all_1s_e = grp_vary['percs_all_1s_e'][()]
                percs_all_1s_h = grp_vary['percs_all_1s_h'][()]
                percs_all_3s_e = grp_vary['percs_all_3s_e'][()]
                percs_all_3s_h = grp_vary['percs_all_3s_h'][()]
                results_fine[label][vary] = (
                    param_values,
                    medians_e, p16_e, p84_e, p3low_e, p3high_e,
                    medians_h, p16_h, p84_h, p3low_h, p3high_h,
                    percs_all_1s_e, percs_all_1s_h, percs_all_3s_e, percs_all_3s_h
                )

def hull_points(x, y, length_threshold=0.03):
    pts = np.column_stack([x, y])
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return np.empty((0, 2))
    idx = concave_hull_indexes(pts, length_threshold=length_threshold)
    hull = pts[idx]
    if len(hull) > 3:
        tck, _ = splprep(hull.T, s=0.0005, per=True)
        u = np.linspace(0, 1.0, 50)
        hull = np.column_stack(splev(u, tck))
    return hull

def make_envelope(x_data, y_data, colour, xlim=(0.25, 1), ylim=(0.1,0.5)):   
    hull = hull_points(x_data, y_data)
    return hv.Polygons(
        [{('x', 'y'): hull}]).opts(
        color=colour, alpha=0.5, line_color=colour,
        width=600, height=600,
        xlim=xlim, ylim=ylim
    )

def colour_for_profile(idx, results_list, base_colour='lightsteelblue'):
    max_vals = np.array([np.nanmin(10**(r[2]-r[1])) for r in results_list])
    max_vals = np.nan_to_num(max_vals, nan=-np.inf)
    N = len(max_vals)
    if N == 0:
        return 'black'
    order_desc = np.argsort(max_vals)
    shades = np.linspace(0.98, 0.12, N)
    shade_for_index = np.empty(N)
    shade_for_index[order_desc] = shades
    s = float(np.clip(shade_for_index[idx], 0, 1))
    white = np.array([1., 1., 1.])
    target = np.array(mpl.colors.to_rgb(base_colour))
    rgb = white*(1-s) + target*s
    return (*rgb, 1.0)

def find_nearest_index(params, target):
    """Find index of nearest parameter pair (params and target must be in same units)."""
    arr = np.array(params, dtype=float)
    scale = np.ptp(arr, axis=0)
    scale[scale == 0] = 1.0
    dist = np.sqrt(np.sum(((target - arr)/scale)**2, axis=1))
    return int(np.nanargmin(dist))


# inset plot functions
def smf(M, M_star0=2e11, alpha=-1.4, m_max=10**12.2):
    phi_star = 2e-3
    phi = phi_star * (M / M_star0)**(alpha + 1) * np.exp(-M / M_star0)
    phi[M > m_max] = 0
    return phi

def smf_inset(alpha, mstar):
    """Accept mstar as log10(M★) from slider and convert to linear for SMF calculation."""
    M_star0 = 10**mstar
    M_grid = np.logspace(9, 12, 25)
    vals = np.log10(smf(M_grid, M_star0=M_star0, alpha=alpha))
    return hv.Curve((np.log10(M_grid), vals)).opts(
        width=300, height=300, line_color='k',
        xlabel='log10(M★)', ylabel='log10(φ)',
        xlim=(9, 12), ylim=(-6, -1.5), show_frame=True,
        tools=['hover'], active_tools=[]
    )

def p_eta_pdf(eta_vals, a=2.05, b=1.90):
    return scipy.stats.beta(a=a, b=b).pdf(eta_vals)

def p_eta_inset(alpha, beta):
    eta = np.linspace(0, 1, 200)
    pdf = p_eta_pdf(eta, a=alpha, b=beta)
    return hv.Curve((eta, pdf)).opts(
        width=300, height=300, line_color='k',
        xlabel='η', ylabel='p(η)',
        xlim=(0, 1), ylim=(0, 2), show_frame=True,
        tools=['hover'], active_tools=[]
    )

# general plotting function for envelope, single profile and inset plot
def make_interactive_plot(results_lists, envelope, inset_func, param_names, defaults, base_colour, log_params):

    raw_params = np.asarray(results_lists[0][0], dtype=float)
    # convert to log10 where asked to
    display_params = raw_params.copy()
    for i, is_log in enumerate(log_params):
        if is_log:
            display_params[:, i] = np.log10(display_params[:, i])

    xs = [r[1] for r in results_lists]
    ys = [r[6] for r in results_lists]
    x_errs = [(r[3] - r[2]) / 2. for r in results_lists]
    y_errs = [(r[8] - r[7]) / 2. for r in results_lists]

    def plot_fn(**kwargs):
        # kwargs are slider values
        target = np.array([kwargs[k] for k in param_names], dtype=float)
        idx = find_nearest_index(display_params, target)

        ebars = []
        
        for x, y, x_e, y_e in zip(xs, ys, x_errs, y_errs):

            ebar_x = hv.ErrorBars((x[idx], y[idx], x_e[idx]), horizontal=True).opts(
                color='k', width=600, height=600, tools=[], active_tools=[]
            )

            ebar_y = hv.ErrorBars((x[idx], y[idx], y_e[idx]), horizontal=False).opts(
                color='k', width=600, height=600, tools=[], active_tools=[],
                xlabel='<ɛ>* / <ɛ>DM', ylabel='<h>* / <h>DM',
            )

            ebars.append(ebar_x * ebar_y)

        inset = inset_func(**{k: kwargs[k] for k in param_names})

        label_1 = hv.Text(0.43, 0.52, 'All').opts(
            text_align='left', text_baseline='top', text_color='black', text_font_size='18pt'
        )
        label_2 = hv.Text(0.76, 0.52, 'r > 100 kpc').opts(
            text_align='left', text_baseline='top', text_color='black', text_font_size='18pt'
        )

        return ((envelope[0] * ebars[0] * label_1) + (envelope[1] * ebars[1] * label_2) + inset).opts(shared_axes=False)


    # build sliders from display_params
    sliders = []
    for i, pname in enumerate(param_names):
        options = np.unique(display_params[:, i])
        # filter nan if any
        options = [float(x) for x in options[np.isfinite(options)]]
        if len(options) == 0:
            options = [defaults[i]]
        slider = pn.widgets.DiscreteSlider(name=pname, options=options, value=defaults[i])
        sliders.append(slider)

    bound_plot = pn.bind(plot_fn, **{s.name: s for s in sliders})
    return pn.Row(pn.Column(*sliders), bound_plot)

cases = {
    'smf': dict(
        results_key='smf',
        inset_func=smf_inset,
        param_names=['alpha', 'mstar'],
        defaults=[-1.3, np.log10(2e11)],
        base_colour='darkorange',
        log_params=[False, True],
    ),
    'p_eta': dict(
        results_key='p_eta',
        inset_func=p_eta_inset,
        param_names=['alpha', 'beta'],
        defaults=[2.05, 1.90],
        base_colour='teal',
        log_params=[False, False],
    ),
}

for label, cfg in cases.items():

    envs = []
    results_lists = []
    for selection, xlim, ylim in zip(['all', 'icl'],
                                       [(0.4, 1), (0.75, 1)],
                                       [(0.1, 0.55), (0.1, 0.55)]):
        results_list = results_fine[selection][cfg['results_key']]

        x_1sigma = results_list[11].flatten()
        y_1sigma = results_list[12].flatten()

        x_3sigma = results_list[13].flatten()
        y_3sigma = results_list[14].flatten()

        env = make_envelope(x_1sigma, y_1sigma, cfg['base_colour'], xlim=xlim, ylim=ylim)
        #env_2 = make_envelope(x_3sigma, y_3sigma, cfg['base_colour'])
        #envs.append(env * env_2)
        envs.append(env)
        results_lists.append(results_list)

    panel = make_interactive_plot(
        results_lists,
        envs,
        cfg['inset_func'],
        cfg['param_names'],
        cfg['defaults'],
        cfg['base_colour'],
        cfg['log_params']
    )

    caption_text = (
    "<p style='font-size:14px; color:black; text-align:center;'>"
    "Figure: Interactive plot showing parameter dependence of the ratio of the average stellar and DM orbital energies vs the ratio of the"
    " average stellar and DM angular momenta for all stripped particles (left) and stripped particles at radii beyond 100 kpc (right)."
    "Use sliders to explore the parameter space."
    "</p>"
    )

    caption = pn.pane.HTML(caption_text)
    layout = pn.Column(panel, caption)

    fname = f'./plots/interactive_e_am_{label}.html'
    layout.save(fname, embed=True)
