
print("Loading libs...")
import xarray as xr
import numpy as np
import pandas as pd
import argparse
import os.path
import tool_fig_config
import scipy
import scipy.stats
import cmocean
print("Done")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--model-versions', type=str, nargs="+", help='Input directory.', required=True)
parser.add_argument('--varset',  type=str, help='Input directory.', default="surf_inst")
parser.add_argument('--varname', type=str, help='Input directory.', default="mean_sea_level_pressure")
parser.add_argument('--year-rng', type=int, nargs=2, help='Range of years', required=True)
parser.add_argument('--months', type=int, nargs="+", help='Month to be processed.', required=True)
parser.add_argument('--level', type=int, help='Selected level if data is 3D.', default=None)
parser.add_argument('--obs-threshold', type=float, help='Month to be processed.', required=True)
parser.add_argument('--offset-far', type=float, help='Month to be processed.', default=-0.05)
parser.add_argument('--offset-hr',  type=float, help='Month to be processed.', default=0)
parser.add_argument('--ROC-thresholds', type=float, nargs="+", help='Month to be processed.', required=True)
parser.add_argument('--labeled-ROC-thresholds', type=float, nargs="+", help='Month to be processed.', default=[])
parser.add_argument('--lead-pentads', type=int, nargs="+", help='Pentad to be processed.', required=True)
parser.add_argument('--days-per-pentad', type=int, help='Pentad to be processed.', required=True)
parser.add_argument('--output', type=str, help='Output directory.', default="")
parser.add_argument('--region', type=str, help='Plot range of latitude', required=True)

parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

print(args)

data = dict()

ROC_thresholds = np.array(args.ROC_thresholds)

labeled_ROC_thresholds_idx = []

for i, labeled_ROC_threshold in enumerate(args.labeled_ROC_thresholds):
    idx = np.argmax(ROC_thresholds == labeled_ROC_threshold)
    labeled_ROC_thresholds_idx.append(idx)

print("idx found: ", labeled_ROC_thresholds_idx)


for model_version in args.model_versions:
    
    print("# model_version : ", model_version)

    filenames = []
    for year in list(range(args.year_rng[0], args.year_rng[1]+1)):

        for month in args.months:
            filenames.append(
                os.path.join(args.input_dir, model_version, "ECCC-S2S_regional_{model_version:s}_{varset:s}::{varname:s}_{yyyymm:s}.nc".format(
                    model_version = model_version,
                    varset  = args.varset,
                    varname = args.varname,
                    yyyymm = pd.Timestamp(year=year, month=month, day=1).strftime("%Y-%m"),
                ))
            )

    print("Reading to load the following files:")
    print(filenames)
    ds = xr.open_mfdataset(filenames, concat_dim="start_time", combine="nested")
    print(ds)

    print(args.region)
    ds = ds.sel(region=str(args.region))

    ds_varnames = dict(
        ECCC  = "ECCC_%s" % args.varname,
        ERA5  = "ERA5_%s" % args.varname,
    )

    if "level" in ds[ds_varnames["ECCC"]].dims:

        if args.level is None:
            
            raise Exception("Data is 3D but `--level` is not given.")

        print("Selecting level = %d" % (args.level,))
        ds = ds.sel(level=args.level)


    numbers = ds.coords["number"]

    empty = np.zeros((len(args.lead_pentads), len(ROC_thresholds),))
    cnt = dict(
        hit = empty.copy(),
        miss = empty.copy(),
        false_alarm = empty.copy(),
        correct_rejection = empty.copy(),
    )

    for st, start_time in enumerate(ds.coords["start_time"]):

        print("Do start_time = ", start_time.to_numpy())

        for lp, lead_pentad in enumerate(args.lead_pentads):
        
            lead_time_rng = np.array([lead_pentad, lead_pentad+1]) * args.days_per_pentad
            _ds = ds.isel(start_time=st, lead_time=slice(*lead_time_rng)).mean(dim="lead_time")
            #print(_ds) 
            ECCC_data = _ds[ds_varnames["ECCC"]].to_numpy().mean()
            ERA5_data = _ds[ds_varnames["ERA5"]].to_numpy()
            
            for j, threshold in enumerate(ROC_thresholds):
                
                obs_detected = ERA5_data >= args.obs_threshold

                ECCC_detected = ECCC_data >= threshold
                     
                if obs_detected:

                    if ECCC_detected:
                        cnt["hit"][lp, j] += 1

                    else:
                        cnt["miss"][lp, j] += 1

                else:
                        
                    if ECCC_detected:
                        cnt["false_alarm"][lp, j] += 1

                    else:
                        cnt["correct_rejection"][lp, j] += 1

                """                
                for p, number in enumerate(numbers):
                    
                    ECCC_detected = ECCC_data[p] >= threshold
                     
                    if obs_detected:

                        if ECCC_detected:
                            cnt["hit"][lp, j] += 1
    
                        else:
                            cnt["miss"][lp, j] += 1

                    else:
                            
                        if ECCC_detected:
                            cnt["false_alarm"][lp, j] += 1
    
                        else:
                            cnt["correct_rejection"][lp, j] += 1
                """
               
    hit_rate         = cnt["hit"]         / ( cnt["hit"]         + cnt["miss"]              )
    false_alarm_rate = cnt["false_alarm"] / ( cnt["false_alarm"] + cnt["correct_rejection"] )

    print(hit_rate.shape)
    print(args.lead_pentads)
    data[model_version]  = xr.Dataset(
        data_vars=dict(
            hit_rate = (["lead_pentad", "threshold"], hit_rate),
            false_alarm_rate = (["lead_pentad", "threshold"], false_alarm_rate),
        ),
        coords=dict(
            lead_pentad=args.lead_pentads,
            threshold=ROC_thresholds,
        ),
        attrs=dict(
            description="S2S forecast data ROC.",
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


nrow = 1
ncol = len(args.lead_pentads)


figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 3,
    h = 3,
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

fig.suptitle("ROC Curve %04d-%04d month=%s" % (
        args.year_rng[0],
        args.year_rng[1],
        ", ".join([ "%02d" % m for m in args.months]),
), size=20)

for i, lead_pentad in enumerate(args.lead_pentads):

    _ax = ax[0, i]

    _ax.set_title("lead_pentad=%d" % (
        lead_pentad,
    ), size=15)
    
    _ax.plot(
        [0, 1],
        [0, 1],
        'k--'
    ) 

   
    for m, model_version in enumerate(args.model_versions):

        d = data[model_version].sel(lead_pentad=lead_pentad)
 
       
        _ax.plot(
            d["false_alarm_rate"],
            d["hit_rate"],
            label=model_version,
            marker='o',
            markersize=5,
        )


        if m == 0:

            for idx in labeled_ROC_thresholds_idx:

                _ax.text(d["false_alarm_rate"][idx] + args.offset_far , d["hit_rate"][idx] + args.offset_hr, "%d" % ( args.ROC_thresholds[idx]), va="center", ha="center")


    _ax.legend()



if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)

print("Finished.")

