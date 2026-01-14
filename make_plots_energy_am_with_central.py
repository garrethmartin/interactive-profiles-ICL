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
                # keep as dict (as you already do)
                results_fine[label][vary] = {key: grp_vary[key][()] for key in grp_vary.keys()}

            else:
                # existing fields
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

                # extra fields needed for central combination
                m_stripped_dm   = grp_vary['m_stripped_dm'][()]
                m_total_dm      = grp_vary['m_total_dm'][()]
                m_stripped_star = grp_vary['m_stripped_star'][()]
                m_total_star    = grp_vary['m_total_star'][()]
                avg_e_star      = grp_vary['avg_e_star'][()]
                avg_e_dm        = grp_vary['avg_e_dm'][()]
                avg_h_star      = grp_vary['avg_h_star'][()]
                avg_h_dm        = grp_vary['avg_h_dm'][()]
                most_massive_sat = grp_vary['most_massive_sat'][()]

                results_fine[label][vary] = (
                    param_values,
                    medians_e, p16_e, p84_e, p3low_e, p3high_e,
                    medians_h, p16_h, p84_h, p3low_h, p3high_h,
                    percs_all_1s_e, percs_all_1s_h, percs_all_3s_e, percs_all_3s_h,
                    # appended extras
                    m_stripped_dm, m_total_dm, m_stripped_star, m_total_star,
                    avg_e_star, avg_e_dm, avg_h_star, avg_h_dm,
                    most_massive_sat
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

def combine_with_central(label, fn, results_fine, radial_h5="radial_profiles.hdf5"):
    """
    Returns a results_list tuple with medians/p16/p84 and percs_all_* replaced by (central+accreted).
    """

    r = results_fine[label][fn]

    # unpack
    param_values = r[0]

    # accreted-only percentiles that define the scatter
    percs1_e = np.asarray(r[11])
    percs1_h = np.asarray(r[12])
    percs3_e = np.asarray(r[13])
    percs3_h = np.asarray(r[14])

    # extras needed for central weighting
    m_stripped_dm   = r[15]
    m_total_dm      = r[16]
    m_stripped_star = r[17]
    m_total_star    = r[18]
    avg_e_star      = r[19]
    avg_e_dm        = r[20]
    avg_h_star      = r[21]
    avg_h_dm        = r[22]
    most_massive_sat = r[23]

    with h5py.File(radial_h5, "r") as f:
        msat_central = f['msat_merger'][()]
        mcentral_stars = f['mcentral_stars'][()]
        mcentral_dm = f['mcentral_dm'][()]

        if label == "icl":
            # ICL-weighted central fractions
            w_star = f['weight_stars_central_icl'][()]
            w_dm   = f['weight_dm_central_icl'][()]
            e_dm_c  = f['e_dm_mean_icl'][()]
            e_st_c  = f['e_stars_mean_icl'][()]
            h_dm_c  = f['am_dm_mean_icl'][()]
            h_st_c  = f['am_stars_mean_icl'][()]
        else:
            e_dm_c  = f['e_dm_mean'][()]
            e_st_c  = f['e_stars_mean'][()]
            h_dm_c  = f['am_dm_mean'][()]
            h_st_c  = f['am_stars_mean'][()]

    # sort for interp
    sidx = np.argsort(msat_central)
    x = msat_central[sidx]
    e_dm_c = e_dm_c[sidx]; e_st_c = e_st_c[sidx]
    h_dm_c = h_dm_c[sidx]; h_st_c = h_st_c[sidx]

    # interpolate central properties at the cluster’s likely most massive satellite
    msat_query = np.asarray(most_massive_sat)
    cluster_e_dm    = np.interp(msat_query, x, e_dm_c)
    cluster_e_stars = np.interp(msat_query, x, e_st_c)
    cluster_h_dm    = np.interp(msat_query, x, h_dm_c)
    cluster_h_stars = np.interp(msat_query, x, h_st_c)

    if label == "icl":
        w_star = w_star[sidx]
        w_dm   = w_dm[sidx]
        star_w = np.interp(msat_query, x, w_star)
        dm_w   = np.interp(msat_query, x, w_dm)
        mcentral_stars_eff = mcentral_stars * star_w
        mcentral_dm_eff    = mcentral_dm * dm_w
    else:
        mcentral_stars_eff = np.full_like(msat_query, float(mcentral_stars))
        mcentral_dm_eff    = np.full_like(msat_query, float(mcentral_dm))

    # satellite contributions scaled to central DM mass (as in your function)
    sat_stars_stripped = (m_stripped_star / m_total_dm) * mcentral_dm
    sat_dm_stripped    = (m_stripped_dm   / m_total_dm) * mcentral_dm

    # mass-weighted combined means
    e_star_comb = (mcentral_stars_eff * cluster_e_stars + avg_e_star * sat_stars_stripped) / (mcentral_stars_eff + sat_stars_stripped)
    e_dm_comb   = (mcentral_dm_eff    * cluster_e_dm    + avg_e_dm   * sat_dm_stripped)    / (mcentral_dm_eff    + sat_dm_stripped)
    h_star_comb = (mcentral_stars_eff * cluster_h_stars + avg_h_star * sat_stars_stripped) / (mcentral_stars_eff + sat_stars_stripped)
    h_dm_comb   = (mcentral_dm_eff    * cluster_h_dm    + avg_h_dm   * sat_dm_stripped)    / (mcentral_dm_eff    + sat_dm_stripped)

    # combined *median* ratios
    e50 = e_star_comb / e_dm_comb
    h50 = h_star_comb / h_dm_comb

    e_off_1s = percs1_e - percs1_e[:, [1]]
    h_off_1s = percs1_h - percs1_h[:, [1]]
    e_off_3s = percs3_e - percs3_e[:, [1]]
    h_off_3s = percs3_h - percs3_h[:, [1]]

    percs1_e_comb = e50[:, None] + e_off_1s
    percs1_h_comb = h50[:, None] + h_off_1s
    percs3_e_comb = e50[:, None] + e_off_3s
    percs3_h_comb = h50[:, None] + h_off_3s

    # extract p16/p50/p84 for errorbars
    p16_e, med_e, p84_e = percs1_e_comb[:, 0], percs1_e_comb[:, 1], percs1_e_comb[:, 2]
    p16_h, med_h, p84_h = percs1_h_comb[:, 0], percs1_h_comb[:, 1], percs1_h_comb[:, 2]

    # rebuild tuple in the same structure your plot code expects
    r_new = (
        param_values,
        med_e, p16_e, p84_e, r[4], r[5],          # keep 3σ bounds if you still want them; or recompute similarly
        med_h, p16_h, p84_h, r[9], r[10],
        percs1_e_comb, percs1_h_comb, percs3_e_comb, percs3_h_comb,
        # append extras unchanged (so you can still use them later)
        *r[15:]
    )
    return r_new

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

        label_1 = hv.Text(0.2, 0.52, 'All').opts(
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

def make_interactive_plot_two_variants(
    results_by_panel,
    envelopes_by_panel,
    inset_func,
    param_names,
    defaults,
    base_colour,
    log_params,
    xlim_by_panel=((0.15, 1), (0.75, 1)),
    ylim_by_panel=((0.1, 0.55), (0.1, 0.55)),
):

    raw_params = np.asarray(results_by_panel[0]["accreted"][0], dtype=float)
    display_params = raw_params.copy()
    for i, is_log in enumerate(log_params):
        if is_log:
            display_params[:, i] = np.log10(display_params[:, i])

    # build sliders
    sliders = []
    for i, pname in enumerate(param_names):
        options = np.unique(display_params[:, i])
        options = [float(x) for x in options[np.isfinite(options)]]
        if len(options) == 0:
            options = [defaults[i]]
        sliders.append(pn.widgets.DiscreteSlider(name=pname, options=options, value=defaults[i]))

    # helper to draw one variant (accreted vs central) for one panel
    def _draw_variant(panel_idx, variant, idx):
        r = results_by_panel[panel_idx][variant]

        # NOTE: your code uses xs = r[1], ys = r[6] etc
        x = r[1][idx]
        y = r[6][idx]
        x_e = (r[3] - r[2])[idx] / 2.0
        y_e = (r[8] - r[7])[idx] / 2.0

        # styling: accreted-only visually lighter/dashed
        if variant == "accreted":
            a = 0.25
            line_dash = "dashed"
            ebar_col = "grey"
        else:
            a = 0.5
            line_dash = "solid"
            ebar_col = "k"

        env = envelopes_by_panel[panel_idx][variant].opts(alpha=a, line_dash=line_dash)

        ebar_x = hv.ErrorBars((x, y, x_e), horizontal=True).opts(
            color=ebar_col, width=600, height=600, tools=[], active_tools=[]
        )
        ebar_y = hv.ErrorBars((x, y, y_e), horizontal=False).opts(
            color=ebar_col, width=600, height=600, tools=[], active_tools=[],
            xlabel='<ɛ>* / <ɛ>DM', ylabel='<h>* / <h>DM'
        )
        # add a visible centre marker for the accreted-only case

        return env * ebar_x * ebar_y

    def plot_fn(**kwargs):
        target = np.array([kwargs[k] for k in param_names], dtype=float)
        idx = find_nearest_index(display_params, target)

        panels = []
        titles = ["All", "r > 100 kpc"]
        for pidx in [0, 1]:
            # overlay central+accreted on top of accreted-only
            overlay = (
                _draw_variant(pidx, "accreted", idx) *
                _draw_variant(pidx, "central", idx)
            ).opts(
                xlim=xlim_by_panel[pidx],
                ylim=ylim_by_panel[pidx],
                width=600, height=600
            )

            title = hv.Text(
                xlim_by_panel[pidx][0] + 0.02*(xlim_by_panel[pidx][1]-xlim_by_panel[pidx][0]),
                ylim_by_panel[pidx][1] - 0.02*(ylim_by_panel[pidx][1]-ylim_by_panel[pidx][0]),
                titles[pidx]
            ).opts(text_align='left', text_baseline='top', text_color='black', text_font_size='18pt')

            panels.append(overlay * title)

        inset = inset_func(**{k: kwargs[k] for k in param_names})

        # left+right + inset
        return (panels[0] + panels[1] + inset).opts(shared_axes=False)

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

variants = ["accreted", "central"]

for label, cfg in cases.items():
    envs = []          # per panel: list of dicts {variant: envelope}
    results_by_panel = []  # per panel: dict {variant: results_list}

    for selection, xlim, ylim in zip(
        ['all', 'icl'],
        [(0.15, 1), (0.75, 1)],
        [(0.1, 0.55), (0.1, 0.55)]
    ):
        # accreted-only (existing)
        r_accreted = results_fine[selection][cfg['results_key']]

        # accreted+central
        r_central = combine_with_central(
            selection, cfg['results_key'], results_fine, radial_h5="radial_profiles.hdf5"
        )

        # envelopes for each variant
        env_map = {}
        for vname, r in [("accreted", r_accreted), ("central", r_central)]:
            x_1sigma = r[11].flatten()
            y_1sigma = r[12].flatten()
            env_map[vname] = make_envelope(x_1sigma, y_1sigma, cfg['base_colour'], xlim=xlim, ylim=ylim)

        envs.append(env_map)
        results_by_panel.append({"accreted": r_accreted, "central": r_central})

    panel = make_interactive_plot_two_variants(
        results_by_panel=results_by_panel,  # [all_panel, icl_panel]
        envelopes_by_panel=envs,            # [env_map_all, env_map_icl]
        inset_func=cfg['inset_func'],
        param_names=cfg['param_names'],
        defaults=cfg['defaults'],
        base_colour=cfg['base_colour'],
        log_params=cfg['log_params'],
    )

    caption_text = (
    "<p style='font-size:14px; color:black; text-align:center;'>"
    "Figure: Interactive envelopes showing the dependence of the ratio of the mass-weighted mean stellar to dark-matter specific orbital "
    "energy, ⟨ε⟩★/⟨ε⟩DM, on the corresponding ratio of specific angular momentum, ⟨h⟩★/⟨h⟩DM. "
    "Results are shown for all stripped material (left) and for material at radii r > 100 kpc (right). "
    "Shaded regions indicate the 1σ envelope spanned by variations in the assumed satellite population, "
    "with lighter, dashed envelopes showing the accreted-only contribution and darker, solid envelopes showing the "
    "combined accreted+central contribution. "
    "Error bars mark the predicted ratios at the selected parameter combination. "
    "Sliders allow interactive exploration of the underlying parameter space."
    "</p>"
    )

    caption = pn.pane.HTML(caption_text)
    layout = pn.Column(panel, caption)

    fname = f'./plots/interactive_e_am_{label}.html'
    layout.save(fname, embed=True)
