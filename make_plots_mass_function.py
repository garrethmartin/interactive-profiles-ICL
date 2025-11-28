import numpy as np
import holoviews as hv
import panel as pn
import h5py
import scipy.stats
import matplotlib as mpl

hv.extension('bokeh')
pn.extension()

# load results
with h5py.File('parameter_exploration_fine.hdf5', 'r') as f:
    results_fine = {}
    for label in f.keys():
        results_fine[label] = {}
        for vary in f[label].keys():
            grp_vary = f[label][vary]
            if vary == 'fiducial':
                results_fine[label][vary] = {key: grp_vary[key][()] for key in grp_vary.keys()}
            else:
                results_fine[label][vary] = (
                    grp_vary['param_values'][()],
                    grp_vary['medians_e'][()],
                    grp_vary['p16_e'][()],
                    grp_vary['p84_e'][()],
                    grp_vary['p3low_e'][()],
                    grp_vary['p3high_e'][()],
                    grp_vary['medians_h'][()],
                    grp_vary['p16_h'][()],
                    grp_vary['p84_h'][()],
                    grp_vary['p3low_h'][()],
                    grp_vary['p3high_h'][()],
                    grp_vary['percs_all_1s_e'][()],
                    grp_vary['percs_all_1s_h'][()],
                    grp_vary['percs_all_3s_e'][()],
                    grp_vary['percs_all_3s_h'][()],
                    grp_vary['stripped_ratio_mean'][()],
                    grp_vary['stellar_stripped_all'][()],
                    grp_vary['m_grids'][()],
                )


def find_nearest_index(params, target):
    """Return index of parameter set closest to target."""
    arr = np.array(params, dtype=float)
    scale = np.ptp(arr, axis=0)
    scale[scale == 0] = 1.0
    dist = np.sqrt(np.sum(((target - arr)/scale)**2, axis=1))
    return int(np.nanargmin(dist))

def compute_shades(all_profiles, base_colour='lightsteelblue'):
    """Precompute colours for each profile based on min value (for ordering)."""
    scalars = np.array([np.nanmin(np.asarray(p)) for p in all_profiles])
    scalars = np.nan_to_num(scalars, nan=-np.inf)
    N = len(scalars)
    if N <= 1:
        return [(*mpl.colors.to_rgb(base_colour), 1.0)]*N
    order = np.argsort(scalars)
    shades = np.linspace(0.98, 0.12, N)
    shade_for_index = np.empty(N)
    shade_for_index[order] = shades
    colours = []
    white = np.array([1., 1., 1.])
    target_rgb = np.array(mpl.colors.to_rgb(base_colour))
    for s in shade_for_index:
        s = float(np.clip(s, 0, 1))
        rgb = white*(1-s) + target_rgb*s
        colours.append((*rgb, 1.0))
    return colours

def make_envelope(y_all, m_log, xlim=(9,12)):
    y_min = np.nanmin(np.where(y_all > 0, y_all, np.nan), axis=0)
    y_max = np.nanmax(np.where(y_all > 0, y_all, np.nan), axis=0)

    area = hv.Area(
        (m_log, y_min, y_max),
        kdims=['log_m'], vdims=['ymin','ymax']
    ).opts(
        fill_color='lightgrey', fill_alpha=0.2,
        line_alpha=0, width=600, height=600,
        xlim=xlim,
        xlabel='log10(M★ / M☉)'
    )
    return area

def smf(M, M_star0=2e11, alpha=-1.4, m_max=10**12.2):
    phi_star = 2e-3
    phi = phi_star * (M / M_star0)**(alpha + 1) * np.exp(-M / M_star0)
    phi[M > m_max] = 0
    return phi

def smf_inset(alpha, mstar):
    M_star0 = 10**mstar
    M_grid = np.logspace(9,12,25)
    vals = np.log10(smf(M_grid, M_star0=M_star0, alpha=alpha))
    return hv.Curve((np.log10(M_grid), vals)).opts(
        width=300, height=300, line_color='k',
        xlabel='log10(M★)', ylabel='log10(φ)',
        xlim=(9,12), ylim=(-6,-1.5),
        show_frame=True, tools=['hover'], active_tools=[]
    )

def p_eta_pdf(eta_vals, a=2.05, b=1.90):
    return scipy.stats.beta(a=a, b=b).pdf(eta_vals)

def p_eta_inset(alpha, beta):
    eta = np.linspace(0,1,200)
    pdf = p_eta_pdf(eta, a=alpha, b=beta)
    return hv.Curve((eta, pdf)).opts(
        width=300, height=300, line_color='k',
        xlabel='η', ylabel='p(η)',
        xlim=(0,1), ylim=(0,2),
        show_frame=True, tools=['hover'], active_tools=[]
    )

def make_interactive_plot(results_lists, envelopes, envelopes_cdf, inset_func, param_names, defaults, log_params, base_colour):
    raw_params = np.asarray(results_lists[0][0], dtype=float)
    display_params = raw_params.copy()
    for i, is_log in enumerate(log_params):
        if is_log:
            display_params[:, i] = np.log10(display_params[:, i])

    xs_all = [r[-1] for r in results_lists]  # m_grids
    ys_all = [r[-2] for r in results_lists]  # stellar_stripped

    # Precompute colours
    curve_colours_all = [compute_shades(y, base_colour) if isinstance(y[0], (list, np.ndarray)) else compute_shades([y], base_colour) for y in ys_all]

    def plot_fn(**kwargs):
        target = np.array([kwargs[k] for k in param_names], dtype=float)
        idx = find_nearest_index(display_params, target)

        inset = inset_func(**{k: kwargs[k] for k in param_names})

        curves = []
        curves_2 = []

        for i, (x, y) in enumerate(zip(xs_all, ys_all)):
            base = dict(
                line_width=3,
                line_color=curve_colours_all[i][idx],
                tools=[], active_tools=[],
                xlabel='log10(M★ / M☉)',
                height=600, width=600
            )

            if i == 0:
                # differential
                diff_opts = dict(
                    **base,
                    ylabel='Stripped stellar mass contribution',
                    ylim=(0, 0.06),
                    line_dash='dashed'
                )
                cum_opts = dict(
                    **base,
                    ylabel='Cumulative stripped stellar mass contribution',
                    ylim=(0, 1.0),
                    line_dash='dashed'
                )
            else:
                # cumulative-only panel
                diff_opts = dict(
                    **base,
                    ylabel='Stripped stellar mass contribution',
                    ylim=(0, 0.06)
                )
                cum_opts = dict(
                    **base,
                    ylabel='Cumulative stripped stellar mass contribution',
                    ylim=(0, 1.0)
                )

            curves.append(
                hv.Curve((x[idx], y[idx])).opts(**diff_opts)
            )

            curves_2.append(
                hv.Curve((x[idx], np.cumsum(y[idx]) / np.sum(y[idx]))).opts(**cum_opts)
            )



        curve_overlay = curves[0] * curves[1]
        curve_overlay_2 = curves_2[0] * curves_2[1]
        env_overlay = envelopes[0] * envelopes[1]
        env_overlay_2 = envelopes_cdf[0] * envelopes_cdf[1]


        return (env_overlay * curve_overlay + env_overlay_2 * curve_overlay_2 + inset).opts(shared_axes=False)

    # Build sliders
    sliders = []
    for i, pname in enumerate(param_names):
        options = np.unique(display_params[:, i])
        options = [float(x) for x in options[np.isfinite(options)]]
        if len(options) == 0:
            options = [defaults[i]]
        sliders.append(pn.widgets.DiscreteSlider(name=pname, options=options, value=defaults[i]))

    bound_plot = pn.bind(plot_fn, **{s.name: s for s in sliders})
    return pn.Row(pn.Column(*sliders), bound_plot)

cases = {
    'smf': dict(
        results_key='smf',
        inset_func=smf_inset,
        param_names=['alpha','mstar'],
        defaults=[-1.3,np.log10(2e11)],
        base_colour='lightsteelblue',
        log_params=[False, True]
    ),
    'p_eta': dict(
        results_key='p_eta',
        inset_func=p_eta_inset,
        param_names=['alpha','beta'],
        defaults=[2.05,1.90],
        base_colour='palegreen',
        log_params=[False, False]
    )
}

for label, cfg in cases.items():
    envelopes = []
    envelopes_cdf = []
    results_lists = []

    for selection, xlim, ylim in zip(['all','icl'], [(9,12),(9,12)], [(0,0.08),(0,0.08)]):
        results_list = results_fine[selection][cfg['results_key']]
        m_grids = results_list[-1]
        stellar_stripped = results_list[-2]

        env = make_envelope(stellar_stripped, m_grids[0])

        cdf_all = np.array([
            np.cumsum(y)/np.sum(y) if np.sum(y) > 0 else np.zeros_like(y)
            for y in stellar_stripped
        ])
        env_cdf = make_envelope(cdf_all, m_grids[0])
        
        envelopes.append(env)
        envelopes_cdf.append(env_cdf)   
        results_lists.append(results_list)

    panel = make_interactive_plot(
        results_lists, envelopes, envelopes_cdf, cfg['inset_func'],
        cfg['param_names'], cfg['defaults'], cfg['log_params'],
        cfg['base_colour']
    )

    caption_text = (
        "<p style='font-size:14px; color:black; text-align:center;'>"
        "Interactive plot showing contribution to the total stripped stellar mass as a function of infall satellite stellar mass. "
        "Curves show all stripped stars (dashed lines) and stars at r > 100 kpc (solid lines). "
        "Use sliders to explore parameter space."
        "</p>"
    )
    caption = pn.pane.HTML(caption_text)
    layout = pn.Column(panel, caption)

    fname = f'./plots/interactive_stripping_{label}.html'
    layout.save(fname, embed=True)
