import xarray as xr
import numpy as np
import scipy
import pandas as pd
import argparse
import os.path
import time_tools


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input', type=str, help='File', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()



ds = xr.open_dataset(args.input)

years  = ds.coords["year"].to_numpy()
months = ds.coords["month"].to_numpy()

plot_lead_weeks = [0, 1, 2]
pvals = []
pval_smalls = []
improveds = []

for lead_week in plot_lead_weeks:
    
    _ds = ds.where((ds.coords["month"] <= 4) | (ds.coords["month"] >= 10)).isel(lead_week=lead_week)
    _pval = _ds["pval"].to_numpy()
    _diff = _ds["mean"].sel(dataset="GEPS6").to_numpy() - _ds["mean"].sel(dataset="GEPS5").to_numpy()
   
    _improved = np.zeros_like(_pval)
    _improved[:] = np.nan

    _pval_small = _pval.copy()
    _pval_small[_pval > 0.1] = np.nan
    
    _improved[_diff < 0] = 1.0 # improve
 
    pvals.append(_pval)
    pval_smalls.append(_pval_small)
    improveds.append(_improved)


print("Loading matplotlib...")

import matplotlib as mplt
if args.no_display:
    mplt.use("Agg")
else:
    mplt.use("TkAgg")


import matplotlib.pyplot as plt
print("done.")

    
fig, ax = plt.subplots(1, len(plot_lead_weeks), squeeze=False, figsize=(10, 4))

for i in plot_lead_weeks:
    
    
    scatter_x = []
    scatter_y = []

    scatter_improved_x = []
    scatter_improved_y = []

    for y, year in enumerate(years):
        for m, month in enumerate(months):
            
            #print(pval_smalls)
            _pval_small = pval_smalls[i][y, m]
            _improved = improveds[i][y, m]
            
            if np.isfinite(_pval_small):
                scatter_y.append(y)
                scatter_x.append(m)

            if np.isfinite(_improved):
                scatter_improved_y.append(y)
                scatter_improved_x.append(m)


    _ax = ax[0, i]

    mappable = _ax.imshow(pvals[i], cmap="gray")
    cb = plt.colorbar(mappable, ax=_ax)

    _ax.scatter(scatter_x, scatter_y, s=20, marker="o", color="yellow")
    _ax.scatter(scatter_improved_x, scatter_improved_y, s=20, marker="x", color="green")

    _ax.set_yticks(np.arange(len(years)), labels=years)
    _ax.set_xticks(np.arange(len(months)), labels=months)

    _ax.set_title("Lead week: %d" % (i,))

if args.output != "":
    print("Saving image file: ", args.output)
    fig.savefig(args.output, dpi=300)
    
if not args.no_display: 
    print("Showing figure...")
    plt.show()


