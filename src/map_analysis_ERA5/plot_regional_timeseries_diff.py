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
parser.add_argument('--models', type=str, nargs=2, help='Input directory.', default=['GEPS5', "GEPS6"])
parser.add_argument('--region', type=str, help='Input directory.', required=True)
parser.add_argument('--ECCC-varset', type=str, help='Input directory.', required=True)
parser.add_argument('--ECCC-varname', type=str, help='Input directory.', required=True)
parser.add_argument('--title', type=str, help='Input directory.', default="")
parser.add_argument('--lead-pentads', type=int, nargs="+", help='Input directory.', required=True)
parser.add_argument('--level', type=int, help='Input directory.', default=None)
parser.add_argument('--thumbnail-numbering', type=str, default="abcdefghijklmn")
parser.add_argument('--thumbnail-numbering-skip', type=int, default=0)
parser.add_argument('--add-datastat', action="store_true")
parser.add_argument('--output', type=str, help='Input directory.', default="")
parser.add_argument('--left-most-month', type=int, help='Input directory.', default=10)
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--percentage', action="store_true", help='Plot range of latitude')
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

    ds = xr.open_dataset(input_file).isel(lead_pentad=args.lead_pentads).sel(region=args.region)

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
        label = "$Z$",
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

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 6,
    h = h,
    wspace = 1.0,
    hspace = 0.5,
    w_left = 1.0,
    w_right = 1.0,
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

plot_info = plot_infos[args.ECCC_varname]
x = None


k1 = list(data.keys())[0]
k2 = list(data.keys())[1]

d1 = data[k1]
d2 = data[k2]

factor = plot_info["factor"] if "factor" in plot_info else 1.0
unit   = plot_info["unit"] 

da_monthly = ( d1["monthly"] - d2["monthly"] )
#da_monthly_std = ( d1["monthly_std"] + d2["monthly_std"] ) / 2 / factor
da_monthly_std = ( d1["monthly_std"]**2 + d2["monthly_std"]**2 )**0.5

yr_cnt = d1["yr_cnt"]

if args.percentage:

    print("Plot in percentage.")    
    da_monthly /= d1["monthly"] / 1e2
    da_monthly_std /= d1["monthly"] / 1e2

else:

    da_monthly /= factor    
    da_monthly_std /= factor    

x = np.arange(len(da_monthly.coords["month"]))

for i, lead_pentad in enumerate(args.lead_pentads):

    _ax1.errorbar(
        x+0.05*i,
        da_monthly.isel(lead_pentad=i),
        da_monthly_std.isel(lead_pentad=i) / yr_cnt.isel(lead_pentad=i)**0.5,
        markersize=5, marker="o", label="%d" % (lead_pentad,)
    )

    if args.percentage:
        _ax1.set_ylabel("[ $ \\% $ ]")

    else:
        _ax1.set_ylabel("[ %s ]"  % (unit,))


_ax1.set_title("(%s) %s%s, %s" % (
    args.thumbnail_numbering[args.thumbnail_numbering_skip + 0],
    args.region,
    ", level = %d hPa" % args.level if is_var3D else "",
    plot_info["label"],
))


x_ticklabels = [ "%d" % m for m in da_monthly.coords["month"] ]
for _ax in ax.flatten():
    _ax.legend()
    _ax.set_xticks(x, x_ticklabels)
    _ax.grid()

    if args.percentage:
        _ax.set_ylim([-10, 15])

if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)

print("Finished.")

