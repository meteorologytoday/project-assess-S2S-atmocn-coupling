import xarray as xr
import numpy as np
import pandas as pd
import argparse
import os.path
import tool_fig_config
import scipy
import scipy.stats
import cmocean

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--model-versions', type=str, nargs=2, help='Input directory.', required=True)
parser.add_argument('--varset',  type=str, help='Input directory.', default="surf_inst")
parser.add_argument('--varname', type=str, help='Input directory.', default="mean_sea_level_pressure")
parser.add_argument('--year-rng', type=int, nargs=2, help='Range of years', required=True)
parser.add_argument('--month', type=int, help='Month to be processed.', required=True)
parser.add_argument('--pval-threshold', type=float, help='Month to be processed.', default=0.05)
parser.add_argument('--lead-pentad', type=int, help='Pentad to be processed.', required=True)
parser.add_argument('--output', type=str, help='Output directory.', default="")
parser.add_argument('--lat-rng', type=float, nargs=2, help='Plot range of latitude', default=[-90, 90])
parser.add_argument('--lon-rng', type=float, nargs=2, help='Plot range of latitude', default=[0, 360])
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

print(args)

data = []

for model_version in args.model_versions:
    
    print("# model_version : ", model_version)

    filenames = []
    for year in list(range(args.year_rng[0], args.year_rng[1]+1)):

        filenames.append(
            os.path.join(args.input_dir, model_version, "ECCC-S2S_{model_version:s}_{varset:s}::{varname:s}_{yyyymm:s}.nc".format(
                model_version = model_version,
                varset  = args.varset,
                varname = args.varname,
                yyyymm = pd.Timestamp(year=year, month=args.month, day=1).strftime("%Y-%m"),
            ))
        )

    print("Reading to load the following files:")
    print(filenames)
    ds = xr.open_mfdataset(filenames, concat_dim="start_ym", combine="nested")
    print(ds)
    ds = ds.sel(lead_pentad=args.lead_pentad)

    ds_varnames = dict(
        Emean    = "%s_Emean" % args.varname,
        E2mean   = "%s_E2mean" % args.varname,
    )

    total_Emean  = ds[ds_varnames["Emean"]].weighted(ds["total_cnt"]).mean(dim="start_ym").rename("total_Emean")
    total_E2mean = ds[ds_varnames["E2mean"]].weighted(ds["total_cnt"]).mean(dim="start_ym").rename("total_E2mean")

    total_Evar = total_E2mean - total_Emean ** 2
    #total_Evar = total_Evar.rename("total_Evar")
    print(total_Evar)

    _total_Evar = total_Evar.to_numpy()
    print("Negative total_Evar (possibly due to precision error): ", _total_Evar[_total_Evar < 0])

    print("Fix the small negative ones...")
    _total_Evar[(np.abs(_total_Evar) < 1e-5) & (_total_Evar < 0)] = 0

    print("Negative total_Evar after fixed: ", _total_Evar[_total_Evar < 0])


    total_Estd = np.sqrt(total_Evar)
    total_Estderr = total_Estd / np.sqrt(ds.dims["start_ym"])

    total_Estd = total_Estd.rename("total_Estd")
    total_Estderr = total_Estderr.rename("total_Estderr")

    total_cnt = ds["total_cnt"].sum(dim="start_ym").rename("total_cnt")

    new_ds = xr.merge([
        total_Emean,
        total_Estd,
        total_Estderr,
        total_cnt,
    ])
    
    new_ds = new_ds.where(
        ( new_ds.coords["latitude"] >= args.lat_rng[0] )
        & ( new_ds.coords["latitude"] <= args.lat_rng[1] )
        & ( new_ds.coords["longitude"] >= args.lon_rng[0] )
        & ( new_ds.coords["longitude"] <= args.lon_rng[0] )
        , drop=True
    )

    new_ds = new_ds.mean(dim=["latitude", "longitude"])

    data.append(new_ds)


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

    mslhf = dict(
        shading_levels = np.linspace(-1, 1, 21) * 50,
        contour_levels = np.linspace(0, 1, 11) * 50,
        factor = 1,
        label = "$ H_\\mathrm{lat} $",
        unit  = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),

    msshf = dict(
        shading_levels = np.linspace(-1, 1, 21) * 50,
        contour_levels = np.linspace(0, 1, 11) * 50,
        factor = 1,
        label = "$ H_\\mathrm{sen} $",
        unit  = "$ \\mathrm{W} \\, / \\, \\mathrm{m}^2 $",
    ),


    sst = dict(
        shading_levels = np.linspace(-1, 1, 21) * 1,
        contour_levels = np.linspace(0, 1, 5) * 1,
        factor = 1,
        label = "SST",
        unit  = "$ \\mathrm{K} $",
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

print("done")



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
    subplot_kw=dict(aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)

cmap = cmocean.cm.balance


_ax = ax[0, 0]

plot_info = plot_infos[args.varname]

fig.suptitle("[%s minus %s] %04d-%04d month=%d, lead_pentad=%d" % (args.model_versions[1], args.model_versions[0], args.year_rng[0], args.year_rng[1], args.month, args.lead_pentad), size=20)


_ax.scatter(
    data[0]["total_Eman"],
    data[1]["total_Eman"],
)


"""
mappable = _ax.contourf(
    coords["longitude"], coords["latitude"],
    _shading,
    levels=plot_info["shading_levels"],
    cmap=cmap, 
    extend="both", 
    transform=proj_norm,
)

# Plot the standard deviation
_contour = (data[0]["total_Estd"] + data[1]["total_Estd"]) / 2 / plot_info["factor"]
cs = _ax.contour(coords["longitude"], coords["latitude"], _contour, levels=plot_info["contour_levels"], colors="k", linestyles='-',linewidths=1, transform=proj_norm, alpha=0.8, zorder=10)
_ax.clabel(cs, fmt="%.1f")

#_contour = pval
#cs = _ax.contour(coords["longitude"], coords["latitude"], _contour, levels=[0.1,], colors="k", linestyles='-',linewidths=1, transform=proj_norm, alpha=0.8, zorder=10)
#_ax.clabel(cs, fmt="%.1f")


# Plot the hatch to denote significant data
_dot = np.zeros_like(pval)
#_dot[:] = np.nan

_significant_idx =  (pval < args.pval_threshold) 
_dot[ _significant_idx                 ] = 0.75
_dot[ np.logical_not(_significant_idx) ] = 0.25

cs = _ax.contourf(coords["longitude"], coords["latitude"], _dot, colors='none', levels=[0, 0.5, 1], hatches=[None, "..."], transform=proj_norm)

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
"""

if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)

print("Finished.")

