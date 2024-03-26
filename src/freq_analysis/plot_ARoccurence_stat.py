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

    d = ds["ARoccur_corr"].to_numpy()[0, :]
    ax.plot(ds.coords["week"], d, color=c, label=dataset_name)
    ax.scatter(ds.coords["week"], d, s=20, c=c)

ax.set_ylim([-0.2, 1])
ax.legend()
ax.grid()

ax.set_xlabel("Lead week")
ax.set_ylabel("Correlation")

if not args.no_display:
    print("Showing figure")
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)

print("Finished.")

