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
parser.add_argument('--models', type=str, nargs="+", help='Input directory.', default=['GEPS5', "GEPS6"])
parser.add_argument('--region', type=str, help='Input directory.', required=True)
parser.add_argument('--ECCC-varset', type=str, help='Input directory.', required=True)
parser.add_argument('--ECCC-varname', type=str, help='Input directory.', required=True)
parser.add_argument('--title', type=str, help='Input directory.', required=True)
parser.add_argument('--lead-pentad', type=int, help='Input directory.', required=True)
parser.add_argument('--level', type=int, help='Input directory.', default=None)
parser.add_argument('--thumbnail-numbering', type=str, default="abcdefghijklmn")
parser.add_argument('--thumbnail-numbering-skip', type=int, default=0)
parser.add_argument('--add-datastat', action="store_true")
parser.add_argument('--output', type=str, help='Input directory.', default="")
parser.add_argument('--left-most-month', type=int, help='Input directory.', default=10)
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

print(args)

varname = "%s_E2mean" % (args.ECCC_varname)
    
data = dict()
is_var3D = False
for model in args.models:

    input_file = "{input_dir:s}/{model:s}/ECCC-S2S_region_{model:s}_{ECCC_varset:s}::{ECCC_varname:s}.nc".format(
        input_dir = args.input_dir,
        model = model,
        ECCC_varset = args.ECCC_varset,
        ECCC_varname = args.ECCC_varname,
    )

    ds = xr.open_dataset(input_file).isel(lead_pentad=args.lead_pentad).sel(region=args.region)

    if "level" in  ds[varname].dims:
        is_var3D = True
        ds = ds.sel(level=args.level)


    print(ds)

    gp = ds.groupby("start_ym.month")
     
    da = ds[varname]**0.5
    da_monthly = da.groupby("start_ym.month").mean("start_ym")
    da_monthly_std = da.groupby("start_ym.month").std("start_ym")
    da_total_cnt = ds["total_cnt"].groupby("start_ym.month").sum("start_ym")

    roll = lambda da : da.roll(
        shifts=dict(month= - ( args.left_most_month - 1) ),
        roll_coords=True,
    )

    da_monthly     = roll(da_monthly)
    da_monthly_std = roll(da_monthly_std)
    da_total_cnt   = roll(da_total_cnt)
        
    yr_cnt = roll(gp.count()[varname])

    data[model] = dict(
        rawdata=da,
        monthly=da_monthly,
        monthly_std=da_monthly_std,
        total_cnt = da_total_cnt,
        yr_cnt = yr_cnt,
    )

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

    gh = dict(
        shading_levels = np.linspace(-1, 1, 21) * 50,
        contour_levels = np.linspace(0, 1, 5) * 50,
        factor = 1,
        label = "$Z_{500}$",
        unit  = "$ \\mathrm{m} $",
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

print("done")

from scipy.stats import ttest_ind_from_stats

ncol = 1
nrow = 1
h = [5,]

if args.add_datastat:
    print("Add datastat!!!!!!")

    nrow += 2
    h = h + [1, 1]


figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 6,
    h = h,
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
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)

_ax1 = ax[0, 0]

if args.add_datastat:
    _ax2 = ax[1, 0]
    _ax3 = ax[2, 0]

#fig.suptitle(args.title, size=20)


plot_info = plot_infos[args.ECCC_varname]
x = None

for i, k in enumerate(data.keys()):

    d = data[k]

    factor = plot_info["factor"] if "factor" in plot_info else 1.0
    unit   = plot_info["unit"] 

    da_monthly = d["monthly"] / factor
    da_monthly_std = d["monthly_std"] / factor
    yr_cnt = d["yr_cnt"]

    #_ax.plot(ds_monthly.coords["month"], da_monthly, markersize=5, marker="o")
   
    #print(da_monthly_std.to_numpy())

    x = np.arange(len(da_monthly.coords["month"]))

    _ax1.errorbar(x + 0.1*i, da_monthly, da_monthly_std / yr_cnt**0.5, markersize=5, marker="o", label=k)
    
    if args.add_datastat:
        #_ax.plot(ds.coords["start_ym"], ds["%s_Emean" % (args.varname)])
        _ax2.bar(x + 0.1*i, d["total_cnt"], 0.1, label=k)
        _ax3.bar(x + 0.1*i, yr_cnt, 0.1, label=k)


_ax1.set_ylabel("[ %s ]"  % (unit,))
_ax1.set_title("(%s) %s%s, %s (lead pentad = %d)" % (
    args.thumbnail_numbering[args.thumbnail_numbering_skip + 0],
    args.region,
    ", level = %d hPa" % args.level if is_var3D else "",
    plot_info["label"],
    args.lead_pentad,
))


if args.add_datastat:
    
    _ax2.set_title("(%s) Number of data" % (
        args.thumbnail_numbering[args.thumbnail_numbering_skip + 1],
    ))
    
    _ax3.set_title("(%s) Number of years" % (
        args.thumbnail_numbering[args.thumbnail_numbering_skip + 2],
    ))


    
x_ticklabels = [ "%d" % m for m in da_monthly.coords["month"] ]
for _ax in ax.flatten():
    _ax.legend()
    _ax.set_xticks(x, x_ticklabels)
    _ax.grid()



if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)

print("Finished.")


