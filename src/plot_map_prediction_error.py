import xarray as xr
import numpy as np
import pandas as pd
import argparse
import os.path
import time_tools
import tool_fig_config

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--varname', type=str, help='Input directory.', default="mean_sea_level_pressure")
parser.add_argument('--dataset-name', type=str, help='Input directory.', required=True)
parser.add_argument('--year-rng', type=int, nargs=2, help='Range of years', required=True)
parser.add_argument('--month', type=int, help='Month to be processed.', required=True)
parser.add_argument('--pentad', type=int, help='Pentad to be processed.', required=True)
parser.add_argument('--output', type=str, help='Output directory.', default="")
parser.add_argument('--plot-lat-rng', type=float, nargs=2, help='Plot range of latitude', default=[-90, 90])
parser.add_argument('--plot-lon-rng', type=float, nargs=2, help='Plot range of latitude', default=[0, 360])
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

print(args)

filenames = []
for year in list(range(args.year_rng[0], args.year_rng[1]+1)):

    filenames.append(
        os.path.join(args.input_dir, "fcst_error_{varname:s}_{yyyymm:s}_pentad-{pentad:d}.nc".format(
            varname = args.varname,
            yyyymm = pd.Timestamp(year=year, month=args.month, day=1).strftime("%Y-%m"),
            pentad = args.pentad,
        ))
    )

print(filenames)
ds = xr.open_mfdataset(filenames, concat_dim="time", combine="nested")
print(ds.time)

# The following code is temporary
total_cnts = np.zeros((len(filenames),))
for i, filename in enumerate(filenames):
    _ds = xr.open_dataset(filename)
    total_cnts[i] = _ds.attrs["total_cnt"]


ds_varnames = dict(
    Emean    = "%s_Emean" % args.varname,
    E2mean   = "%s_E2mean" % args.varname,
    Eabsmean = "%s_Eabsmean" % args.varname,
)


ds_mean = ds.mean(dim="time")

total_Emean = np.zeros_like(ds_mean[ds_varnames["Emean"]])
total_E2mean = np.zeros_like(ds_mean[ds_varnames["Emean"]])
for i in range(ds.dims["time"]):
    total_Emean  += total_cnts[i] * ds[ds_varnames["Emean"]].isel(time=i).to_numpy()
    total_E2mean += total_cnts[i] * ds[ds_varnames["E2mean"]].isel(time=i).to_numpy()
    

total_Emean  /= np.sum(total_cnts)
total_E2mean /= np.sum(total_cnts)

total_Evar = total_E2mean - total_Emean ** 2

print("Negative total_Evar (possibly due to precision error): ", total_Evar[total_Evar < 0])

print("Fix the small negative ones...")
total_Evar[(np.abs(total_Evar) < 1e-5) & (total_Evar < 0)] = 0

print("Negative total_Evar after fixed: ", total_Evar[total_Evar < 0])


total_Estd = np.sqrt(total_Evar)
total_Estderr = total_Estd / np.sqrt(ds.dims["time"])

new_ds = xr.Dataset(
    data_vars=dict(
        mean_Emean   = (["lat", "lon"], ds_mean[ds_varnames["Emean"]].to_numpy()),
        mean_E2mean  = (["lat", "lon"], ds_mean[ds_varnames["E2mean"]].to_numpy()),
        total_Emean  = (["lat", "lon"], total_Emean),
        total_E2mean = (["lat", "lon"], total_E2mean),
        total_Estd   = (["lat", "lon"], total_Estd),
        total_Estderr   = (["lat", "lon"], total_Estderr),
    ),
    coords=dict(
        lat = ds_mean.coords["lat"],
        lon = ds_mean.coords["lon"],
    ),
)


ds.close()


plot_infos = dict(
    mean_sea_level_pressure = dict(
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

cmap = mplt.cm.get_cmap("bwr")

cmap.set_over("green")
cmap.set_under("yellow")

_ax = ax[0, 0]

plot_info = plot_infos[args.varname]

fig.suptitle("[%s] %04d-%04d month=%d, pentad=%d" % (args.dataset_name, args.year_rng[0], args.year_rng[1], args.month, args.pentad), size=20)





coords = new_ds.coords

_shading = new_ds["mean_Emean"].to_numpy() / plot_info["factor"]
mappable = _ax.contourf(
    coords["lon"], coords["lat"],
    _shading,
    levels=plot_info["shading_levels"],
    cmap=cmap, 
    extend="both", 
    transform=proj_norm,
)

# Plot the standard deviation
_contour = new_ds["total_Estd"] / plot_info["factor"]
cs = _ax.contour(coords["lon"], coords["lat"], _contour, levels=plot_info["contour_levels"], colors="k", linestyles='-',linewidths=1, transform=proj_norm, alpha=0.8, zorder=10)
_ax.clabel(cs, fmt="%.1f")


# Plot the hatch to denote significant data
_significant = np.abs(new_ds["mean_Emean"].to_numpy()) - new_ds["total_Estderr"].to_numpy()
_dot = np.zeros_like(_significant)
#_dot[:] = np.nan

_significant_idx =  (_significant > 0) 
_dot[ _significant_idx                 ] = 0.75
_dot[ np.logical_not(_significant_idx) ] = 0.25

cs = _ax.contourf(coords["lon"], coords["lat"], _dot, colors='none', levels=[0, 0.5, 1], hatches=[None, "..."], transform=proj_norm)

# Remove the contour lines for hatches 
for _, collection in enumerate(cs.collections):
    collection.set_edgecolor((.2, .2, .2))
    collection.set_linewidth(0.)

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

