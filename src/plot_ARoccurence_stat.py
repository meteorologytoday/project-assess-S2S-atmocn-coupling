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
for i in range(N):
    data[args.dataset_names[i]] = xr.open_dataset(args.input_files[i])


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

ncol=1
nrow=1

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
)

if args.title != "":
    fig.suptitle(args.title)

for dataset_name, ds in data.items():

    ax.plot(ds.coords["week"], ds["ARoccur_corr"], label=dataset_name)

ax.legend()

if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)

print("Finished.")

