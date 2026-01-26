import numpy as np
import holoviews as hv
import panel as pn
import h5py
from holoviews import opts

hv.extension('bokeh')
pn.extension()

# load precomputed contours
results = {}
with h5py.File('precomputed_contours.hdf5', 'r') as f:
    for grp_name in f.keys():
        grp = f[grp_name]
        parts = grp_name.split('_')
        mass_ratio = float(parts[1])
        eta = float(parts[3])

        data = {
            'lim_x': grp.attrs['lim_x'],
            'lim_y': grp.attrs['lim_y'],
            'error_bars': {},
        }

        err_grp = grp['error_bars']
        for comp in ['stars', 'dm']:
            data['error_bars'][comp] = {
                'e': err_grp[comp]['e'][()],
                'h': err_grp[comp]['h'][()]
            }

        for key in ['kde_star_x', 'kde_dm_x', 'kde_star_y', 'kde_dm_y']:
            sg = grp[key]
            data[key] = (sg['x'][()], sg['y'][()])

        for key in ['contour_star', 'contour_dm']:
            sg = grp[key]
            data[key] = (sg['xx'][()], sg['yy'][()], sg['zz'][()])

        results[(mass_ratio, eta)] = data

# load precomputed RGB images
with h5py.File('precomputed_images_spatial.hdf5', 'r') as f_img:
    for grp_name in f_img.keys():
        parts = grp_name.split('_')
        mass_ratio = float(parts[1])
        eta = float(parts[3])
        if (mass_ratio, eta) not in results:
            continue
        grp = f_img[grp_name]
        results[(mass_ratio, eta)]['rgb_image'] = grp['rgb_image'][()]
        results[(mass_ratio, eta)]['h_range'] = grp.attrs['h_range']

#  helper functions
def pad(width, height):
    """Small invisible plot for spacing in layouts."""
    return hv.Curve((np.array([0,1]), np.array([0,0]))).opts(
        width=width, height=height,
        line_alpha=0, xaxis=None, yaxis=None,
        show_legend=False, toolbar=None
    )

def plot_energy_am(mass, eta):
    d = results[(mass, eta)]

    # bounds from saved limits
    xlim = tuple(d['lim_x'])
    ylim = tuple(d['lim_y'])
    bounds = (np.min(xlim), np.min(ylim), np.max(xlim), np.max(ylim))

    # 2D contours
    xx, yy, zz = d['contour_star']
    img_star = hv.Image(np.flipud(zz), bounds=bounds)
    contour_star = hv.operation.contours(img_star, levels=6).opts(
        line_color='darkgreen', line_width=2, alpha=0.6, color_index=None, cmap=None
    )

    xx, yy, zz = d['contour_dm']
    img_dm = hv.Image(np.flipud(zz), bounds=bounds)
    contour_dm = hv.operation.contours(img_dm, levels=6).opts(
        line_color='purple', line_width=2, alpha=0.6, color_index=None, cmap=None
    )

    # error bars
    es_star, hs_star = d['error_bars']['stars']['e'], d['error_bars']['stars']['h']
    es_dm, hs_dm = d['error_bars']['dm']['e'], d['error_bars']['dm']['h']

    stars_err_x = hv.ErrorBars([(es_star[1], hs_star[1], es_star[1]-es_star[0])], horizontal=True).opts(color='darkgreen', line_width=2)
    stars_err_y = hv.ErrorBars([(es_star[1], hs_star[1], hs_star[1]-hs_star[0])], horizontal=False).opts(color='darkgreen', line_width=2)
    dm_err_x = hv.ErrorBars([(es_dm[1], hs_dm[1], es_dm[1]-es_dm[0])], horizontal=True).opts(color='purple', line_width=2)
    dm_err_y = hv.ErrorBars([(es_dm[1], hs_dm[1], hs_dm[1]-hs_dm[0])], horizontal=False).opts(color='purple', line_width=2)

    main_plot = (contour_dm * contour_star * stars_err_x * stars_err_y * dm_err_x * dm_err_y)

    # add text with mass and eta
    text = hv.Text(
        x=1.2e6, y=6.5e5, text=f"Msat/Mhost = {mass}\nη = {eta}"
    ).opts(
        text_font_size='12pt', text_color='black', text_align='left',
        width=600, height=600,
        ylim=(-0.1e5, 7e5), xlim=(1e6, 5e6),
        xlabel='ɛ [kpc km²/s²]', ylabel='h [kpc km/s]',
        tools=[], active_tools=[]
    )
    main_plot = main_plot * text

    # x-marginal
    kx_star_x, ky_star_x = d['kde_star_x']
    kx_dm_x, ky_dm_x = d['kde_dm_x']
    kde_x = (hv.Curve((kx_star_x, ky_star_x)).opts(color='darkgreen', line_width=2) *
             hv.Curve((kx_dm_x, ky_dm_x)).opts(color='purple', line_width=2)).opts(
                 xlim=(1e6, 5e6),
                 width=600, height=100, xaxis=None, yaxis=None, tools=[], active_tools=[]
             )

    # y-marginal
    kx_star_y, ky_star_y = d['kde_star_y']
    kx_dm_y, ky_dm_y = d['kde_dm_y']
    kde_y = (hv.Curve((ky_star_y, kx_star_y)).opts(color='darkgreen', line_width=2) *
             hv.Curve((ky_dm_y, kx_dm_y)).opts(color='purple', line_width=2)).opts(
                 ylim=(-0.1e5, 7e5),
                 width=100, height=600, xaxis=None, yaxis=None, tools=[], active_tools=[]
             )

    # padding
    pad_left_small = pad(width=5, height=5)
    pad_right_small = pad(width=100, height=100)
    pad_left_tall = pad(width=5, height=600)

    # Create RGB image plot if available
    if 'rgb_image' in d:
        rgb_img = d['rgb_image']
        
        # Downsample to half resolution using block averaging for smoother appearance
        h, w, c = rgb_img.shape
        rgb_img_half = rgb_img.reshape(h//2, 2, w//2, 2, c).mean(axis=(1, 3))
        
        rgb_plot = hv.RGB(rgb_img_half).opts(
            width=600, height=600,
            xaxis=None, yaxis=None,
            tools=[], active_tools=[],
            aspect='equal'
        )
        
        # Add spacing elements for RGB column
        pad_top_rgb = pad(width=600, height=100)
        pad_right_rgb = pad(width=100, height=600)
        
        # grid: 4 columns, 2 rows
        elements = [
            pad_left_small, kde_x, pad_right_small, pad_top_rgb,
            pad_left_tall, main_plot, kde_y, rgb_plot
        ]
        layout = hv.Layout(elements).cols(4).opts(opts.Layout(shared_axes=False))
    else:
        elements = [
            pad_left_small, kde_x, pad_right_small,
            pad_left_tall, main_plot, kde_y
        ]
        layout = hv.Layout(elements).cols(3).opts(opts.Layout(shared_axes=False))
    
    return layout

# build sliders
mass_values = sorted({key[0] for key in results.keys()})
eta_values = sorted({key[1] for key in results.keys()})

mass_slider = pn.widgets.DiscreteSlider(name='Msat/Mhost', options=mass_values, value=mass_values[4])
eta_slider = pn.widgets.DiscreteSlider(name='η', options=eta_values, value=eta_values[2])

interactive_plot = pn.bind(plot_energy_am, mass=mass_slider, eta=eta_slider)

# assemble panel with caption
caption_text = (
    "<p style='font-size:16px; color:black; text-align:center;'>"
    "Figure: Interactive plot of the specific orbital energy-angular momentum distribution of stripped stellar and dark matter components for each satellite model and orbit. "
    "The sliders adjust the satellite-to-host mass ratio (Msat/Mhost) and orbital circularity η. "
    "Left: The main panel shows contours of the stellar (green) and dark matter (purple) distributions, with error bars indicating the median and 1σ dispersion. "
    "Marginal distributions in ɛ and h are shown along the horizontal and vertical axes, respectively. "
    "Right: Spatial distribution of the stripped material with stars shown in green and DM shown in purple."
    "</p>"
)
caption = pn.pane.HTML(caption_text)

layout = pn.Column(
    interactive_plot,
    pn.Row(mass_slider, eta_slider),
    caption
)

layout.save('./plots/e_h_dist.html', embed=True)