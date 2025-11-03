import numpy as np
import holoviews as hv
import panel as pn
import matplotlib as mpl
import h5py
import scipy.stats

hv.extension('bokeh')
pn.extension()

# load results
with h5py.File('stripped_density_profiles_rbins_weighted_variations_fine.h5', 'r') as hf:
    results_fine = {}
    for label in hf.keys():
        results_fine[label] = {}
        for key in hf[label].keys():
            grp = hf[label][key]
            if key == 'fiducial':
                results_fine[label]['fiducial'] = (
                    grp['log_rho_dm'][:],
                    grp['log_rho_star'][:],
                )
            else:
                results = []
                for combo_str, combo_grp in grp.items():
                    combo = tuple(map(float, combo_str.split('_')))
                    results.append((combo,
                                    combo_grp['log_rho_dm'][:],
                                    combo_grp['log_rho_star'][:]))
                results_fine[label][key] = results

# helper functions
def make_envelope(y_all, r_log, xlim=(-2, 0), ylim=(-3.6, 0.2)):
    y_min = np.nanmin(np.where(y_all > 0, y_all, np.nan), axis=0)
    y_max = np.nanmax(np.where(y_all > 0, y_all, np.nan), axis=0)
    return hv.Area(
        (r_log, np.log10(y_min), np.log10(y_max)),
        kdims=['log_r'], vdims=['ymin', 'ymax']
    ).opts(
        fill_color='lightgrey', fill_alpha=0.4, line_alpha=0,
        width=600, height=600, xlim=xlim, ylim=ylim,
        xlabel='log10(r / R200)', ylabel='log10(ρ* / ρDM)'
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
def make_interactive_plot(results_list, envelope, inset_func, param_names, defaults, base_colour, log_params):

    raw_params = np.array([r[0] for r in results_list], dtype=float)
    # convert to log10 where asked to
    display_params = raw_params.copy()
    for i, is_log in enumerate(log_params):
        if is_log:
            display_params[:, i] = np.log10(display_params[:, i])

    y_all = np.array([10**(r[2] - r[1]) for r in results_list])
    y_all_log = np.log10(np.where(y_all > 0, y_all, np.nan))

    def plot_fn(**kwargs):
        target = np.array([kwargs[k] for k in param_names], dtype=float)
        idx = find_nearest_index(display_params, target)

        y_profile = y_all_log[idx]
        mask = r_log > -1  # region for slope fitting
        x_fit = r_log[mask]
        y_fit = y_profile[mask]

        # Ensure finite values
        valid = np.isfinite(x_fit) & np.isfinite(y_fit)
        if np.sum(valid) > 2:
            slope, intercept = np.polyfit(x_fit[valid], y_fit[valid], 1)
        else:
            slope, intercept = np.nan, np.nan

        color = colour_for_profile(idx, results_list, base_colour)
        curve = hv.Curve((r_log, y_profile)).opts(
            color=color, line_width=6,
            xlabel='log10(r / R200)', ylabel='log10(ρ* / ρDM)',
            xlim=(-1, 0), ylim=(-3.6, 0.2),
            tools=['hover'], active_tools=[]
        )

        # Add slope annotation
        if np.isfinite(slope):
            # Pick text position slightly above the last valid y in fit region
            x_text = -0.5
            y_text = np.polyval([slope, intercept], x_text) + 0.1
            label = hv.Text(
                x_text, y_text, f"slope = {slope:.2f}"
            ).opts(
                text_align='left',
                text_baseline='bottom',
                text_color='black',
                text_font_size='12pt'
            )
            curve = curve * label

        inset = inset_func(**{k: kwargs[k] for k in param_names})
        return (envelope * curve) + inset


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

R200 = 1112.1
r_bins = np.asarray([11.45775381, 14.79827068, 19.11271778, 24.68504523,
                    31.88198899, 41.1772072 , 53.18245336, 68.68783821,
                    88.71382982, 114.57841457, 147.98383874, 191.12863979,
                    246.8523405 ,318.82232865, 411.77522175, 531.82860172,
                    686.88363619, 887.14508423, 1145.79291019, 1479.84970708])

r = r_bins / R200
r_log = np.log10(r)

cases = {
    'smf': dict(
        results_key='smf',
        inset_func=smf_inset,
        param_names=['alpha', 'mstar'],
        defaults=[-1.3, np.log10(2e11)],
        base_colour='lightsteelblue',
        log_params=[False, True],
    ),
    'p_eta': dict(
        results_key='p_eta',
        inset_func=p_eta_inset,
        param_names=['alpha', 'beta'],
        defaults=[2.05, 1.90],
        base_colour='palegreen',
        log_params=[False, False],
    ),
}

for label, cfg in cases.items():
    results_list = results_fine['all'][cfg['results_key']]
    y_all_case = np.array([10**(r[2] - r[1]) for r in results_list])
    env = make_envelope(y_all_case, r_log)
    panel = make_interactive_plot(
        results_list,
        env,
        cfg['inset_func'],
        cfg['param_names'],
        cfg['defaults'],
        cfg['base_colour'],
        cfg['log_params']
    )
    
    if label == 'smf':
        caption_text = (
        "<p style='font-size:16px; color:black; text-align:center;'>"
        "Figure: Left: Interactive plot of the ratio between stellar and dark-matter radial density profiles predicted by the model for a given infalling satellite population. "
        "Each curve is the superposed profile obtained by integrating individual satellite contributions across the population; the sliders adjust the population parameters (low-mass slope α and characteristic stellar mass M★), which alter the integrated profile. "
        "Right: The corresponding infall galaxy stellar mass function for the current slider settings."
        "</p>"
        )

    if label == 'p_eta':
        caption_text = (
        "<p style='font-size:16px; color:black; text-align:center;'>"
        "Figure: Left: Interactive plot of the ratio between stellar and dark-matter radial density profiles predicted by the model for a given infalling satellite population. "
        "Each curve is the superposed profile obtained by integrating individual satellite contributions across the population; the sliders adjust the population's orbital circularity distribution (α and β), which alter the integrated profile. "
        "Right: The corresponding infall satellite orbital circularity distribution for the current slider settings."
        "</p>"
        )

    caption = pn.pane.HTML(caption_text)
    layout = pn.Column(panel, caption)

    fname = f'./plots/interactive_radial_{label}_with_inset.html'
    layout.save(fname, embed=True)
