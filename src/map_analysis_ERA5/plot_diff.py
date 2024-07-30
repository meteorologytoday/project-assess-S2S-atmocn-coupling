import xarray as xr
import numpy as np
import pandas as pd
import argparse
import os.path
import tool_fig_config
import cmocean

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--files', type=str, nargs=2, help='Input directories. A minus B.', required=True)
parser.add_argument('--varname', type=str, help='Input directory.', default="msl")
parser.add_argument('--title', type=str, help='Input directory.', default="")
parser.add_argument('--output', type=str, help='Output directory.', default="")
parser.add_argument('--plot-lat-rng', type=float, nargs=2, help='Plot range of latitude', default=[-90, 90])
parser.add_argument('--plot-lon-rng', type=float, nargs=2, help='Plot range of latitude', default=[0, 360])
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

print(args)

data = []
for fname in args.files:
    print("Opening dataset: ", fname)
    data.append(xr.open_dataset(fname).isel(time=0))

print("Datasets all opened.")
ds_diff = data[0] - data[1]

plot_infos = dict(

    IWV = dict(
        shading_levels = np.linspace(-1, 1, 21) * 10,
        contour_levels = np.linspace(0, 1, 11) * 10,
        factor = 1.0,
        label = "IWV",
        unit  = "$\\mathrm{kg} / \\mathrm{m}^2$",
    ),

    IVT = dict(
        shading_levels = np.linspace(-1, 1, 21) * 50,
        contour_levels = np.linspace(0, 1, 11) * 40,
        factor = 1.0,
        label = "IVT",
        unit  = "$\\mathrm{kg} / \\mathrm{m} / \\mathrm{s}$",
    ),

    IVT_x = dict(
        shading_levels = np.linspace(-1, 1, 21) * 50,
        contour_levels = np.linspace(0, 1, 11) * 40,
        factor = 1.0,
        label = "$\\mathrm{IVT}_x$",
        unit  = "$\\mathrm{kg} / \\mathrm{m} / \\mathrm{s}$",
    ),



    IVT_y = dict(
        shading_levels = np.linspace(-1, 1, 21) * 50,
        contour_levels = np.linspace(0, 1, 11) * 40,
        factor = 1.0,
        label = "$\\mathrm{IVT}_y$",
        unit  = "$\\mathrm{kg} / \\mathrm{m} / \\mathrm{s}$",
    ),




    u10 = dict(
        shading_levels = np.linspace(-1, 1, 21) * 5,
        contour_levels = np.linspace(0, 1, 11) * 10,
        factor = 1.0,
        label = "$ u_\\mathrm{10m} $",
        unit  = "m / s",
    ),

    v10 = dict(
        shading_levels = np.linspace(-1, 1, 21) * 5,
        contour_levels = np.linspace(0, 1, 11) * 10,
        factor = 1.0,
        label = "$ v_\\mathrm{10m} $",
        unit  = "m / s",
    ),



    msl = dict(
        shading_levels = np.linspace(-1, 1, 21) * 5,
        contour_levels = np.linspace(0, 1, 11) * 20,
        factor = 1e2,
        label = "$ P_\\mathrm{sfc} $",
        unit  = "hPa",
    ),

)

print("Loading matplotlib...")

import matplotlib as mplt
if args.no_display:
    mplt.use("Agg")
else:
    mplt.use("TkAgg")


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

print("done")

from scipy.stats import ttest_ind_from_stats


cent_lon = 180.0

plot_lon_l = args.plot_lon_rng[0]
plot_lon_r = args.plot_lon_rng[1]
plot_lat_b = args.plot_lat_rng[0]
plot_lat_t = args.plot_lat_rng[1]

proj = ccrs.PlateCarree(central_longitude=cent_lon)
proj_norm = ccrs.PlateCarree()

ncol = 1
nrow = 1


figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 10,
    h = 5,
    wspace = 1.0,
    hspace = 0.5,
    w_left = 1.0,
    w_right = 2.2,
    h_bottom = 1.0,
    h_top = 1.0,
    ncol = ncol,
    nrow = nrow,
)


fig, ax = plt.subplots(
    nrow, ncol,
    figsize=figsize,
    subplot_kw=dict(projection=proj, aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)

cmap = cmocean.cm.balance

_ax = ax[0, 0]

plot_info = plot_infos[args.varname]

fig.suptitle(args.title, size=20)

coords = ds_diff.coords

_shading = ds_diff[args.varname].to_numpy() / plot_info["factor"]
mappable = _ax.contourf(
    coords["longitude"], coords["latitude"],
    _shading,
    levels=plot_info["shading_levels"],
    cmap=cmap, 
    extend="both", 
    transform=proj_norm,
)

for __ax in [_ax, ]: 

    gl = __ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')

    gl.xlabels_top   = False
    gl.ylabels_right = False

    #gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    #gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
    #gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])
    
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    gl.ylabel_style = {'size': 12, 'color': 'black'}

    __ax.set_global()
    #__ax.gridlines()
    __ax.coastlines(color='gray')
    __ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)



cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)

unit_str = "" if plot_info["unit"] == "" else " [ %s ]" % (plot_info["unit"],)
cb.ax.set_ylabel("%s\n%s" % (plot_info["label"], unit_str), size=25)


if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)

print("Finished.")

