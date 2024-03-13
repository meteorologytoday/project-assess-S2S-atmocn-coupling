import xarray as xr
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-file', type=str, help='Input directory.', required=True)
parser.add_argument('--isel-start-time', type=int, help='Input directory.', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--varname', type=str, help='Output filename in png.', default="mean_sea_level_pressure_RMS")
parser.add_argument('--y-rng', type=float, nargs=2, help='Range of y-axis.', default=[None, None])
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

filename = args.input_file


ds = xr.open_dataset(filename)[args.varname]
ds = ds.isel(start_time=args.isel_start_time)

lead_time = ds.coords["lead_time"].to_numpy()/(3600*1e9)
    
print("Loading matplotlib...")
import matplotlib as mplt
mplt.use("TkAgg")


import matplotlib.pyplot as plt
print("done.")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for ens in ds.coords["ens"]:
    _data = ds.sel(ens=ens)
    ax.plot(lead_time, _data)


ax.set_ylabel("RMS [Pa]")
ax.set_xlabel("Lead Time [hr]")
ax.set_title(filename)
ax.set_ylim(args.y_rng)
ax.grid()

if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

