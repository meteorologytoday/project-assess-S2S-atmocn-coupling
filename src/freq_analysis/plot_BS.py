import xarray as xr
import numpy as np
import pandas as pd
import argparse
import os.path
import time_tools
import tool_fig_config

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-files',   type=str, nargs="+", help='Input directory.', required=True)
parser.add_argument('--dataset-names', type=str, nargs="+", help='Input directory.', required=True)
parser.add_argument('--title', type=str, help='Input directory.', default="")
parser.add_argument('--output', type=str, help='Input directory.', default="")
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

print(args)


if len(args.input_files) != len(args.dataset_names):
    raise Exception("Length of input files does not equal to length of dataset names")

N = len(args.input_files)

data = dict()

N_of_categories = None
for i in range(N):
    ds = xr.open_dataset(args.input_files[i])
    data[args.dataset_names[i]] = ds

    if N_of_categories is None:
        N_of_categories = ds.dims["category"]



ds_mean = None
cnt = 0
for k, ds in data.items():

    if cnt == 0:
        ds_mean = ds.copy()

    else:
        ds_mean += ds

    cnt += 1
ds_mean /= cnt



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

ncol = N_of_categories
nrow = 1

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 6,
    h = 4,
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
    subplot_kw=dict(aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
)

print("Figure initiated")
if args.title != "":
    fig.suptitle(args.title)

for i, (dataset_name, ds) in enumerate(data.items()):
        
    c = ["red", "blue"][i]

    for j in range(N_of_categories):    
       
        _ax = ax[j]
        
        BS_ECCC = ds["BS_ECCC"].isel(month_group=0, category=j)
        BS_clim = ds_mean["BS_clim"].isel(month_group=0, category=j)
       
        BSS = 1.0 - BS_ECCC / BS_clim
 
        #_ax.plot(ds.coords["week"], BS_ECCC, color=c, linestyle="solid", label=dataset_name)
        #_ax.plot(ds.coords["week"], BS_clim, color=c, linestyle="dashed", label=dataset_name)
        
        #_ax.scatter(ds.coords["week"], BS_ECCC, s=20, c=c)
        #_ax.scatter(ds.coords["week"], BS_clim, s=20, c=c)
        
        _ax.plot(ds.coords["week"], BSS, color=c, linestyle="solid", label=dataset_name)
        _ax.scatter(ds.coords["week"], BSS, s=20, c=c)

        _ax.set_title("Category : %d" % (j, ))

for _ax in ax.flatten(): 
    _ax.set_ylim([-0.2, 1])
    _ax.legend()
    _ax.grid()

    _ax.set_xlabel("Lead week")
    _ax.set_ylabel("Correlation")

if not args.no_display:
    print("Showing figure")
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)

print("Finished.")

