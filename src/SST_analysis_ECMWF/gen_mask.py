from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import argparse
import traceback
import os
import pretty_latlon



pretty_latlon.default_fmt = "%d"

parser = argparse.ArgumentParser(
                    prog = 'make_ECCC_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

parser.add_argument('--test-input', type=str, required=True)
parser.add_argument('--test-varname', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()
print(args)

print("Test input: ", args.test_input)

ds = xr.open_dataset(args.test_input)
test_da = ds[args.test_varname].isel(start_time=0, lead_time=0).load()

regions = [
    "NPAC_ALL", "NPAC_SOUTH", "NPAC_NORTH", "NPAC_EAST", "NPAC_WEST", "WWRF" 
]


mask = xr.apply_ufunc(np.isfinite, test_da)
base_mask = mask.to_numpy().astype(int)

# Copy is necessary so that the value can be assigned later
mask = mask.expand_dims(
    dim = dict(region=regions),
    axis=0,
).rename("mask").copy()

print(mask)

masks = np.zeros(
    (len(regions), len(mask.coords["latitude"]), len(mask.coords["longitude"]),),
    dtype=int,
)

for i, region in enumerate(regions):
   
    print("Making region: ", region) 
    _mask = mask.sel(region=region).copy()
    lat = _mask.coords["latitude"]
    lon = _mask.coords["longitude"] % 360.0

    if region == "NPAC_ALL":
        
        _mask = _mask.where(
            (lat > 0) &
            (lat <= 60) &
            (lon > 120) &
            (lon <= 250) 
        )

    elif region == "NPAC_SOUTH":
        
        _mask = _mask.where(
            (lat > 0) &
            (lat <= 30) &
            (lon > 120) &
            (lon <= 250) 
        )


    elif region == "NPAC_NORTH":
        
        _mask = _mask.where(
            (lat > 30) &
            (lat <= 60) &
            (lon > 120) &
            (lon <= 250) 
        )

    elif region == "NPAC_EAST":
        
        _mask = _mask.where(
            (lat > 0) &
            (lat <= 60) &
            (lon > 180) &
            (lon <= 250) 
        )

    elif region == "NPAC_WEST":
        
        _mask = _mask.where(
            (lat > 0) &
            (lat <= 60) &
            (lon > 120) &
            (lon <= 180) 
        )
 
    elif region == "WWRF":
        
        _mask = _mask.where(
            (lat > 5) &
            (lat <= 62) &
            (lon > 160) &
            (lon <= 250) 
        )
 
 
    else:
        
        print("Warning: Cannot find code to define region `%s`." % (region,))
        continue 
    _mask = xr.apply_ufunc(np.isfinite, _mask)
    mask[i, :, :] = _mask.to_numpy().astype(int) * base_mask

new_ds = xr.merge([mask, ])

lat = new_ds.coords["latitude"].to_numpy()
print(lat) 
if lat[1] < lat[0]:
    print("Reverse latitude.")
    new_ds = new_ds.isel(latitude=slice(None, None, -1))

new_ds = new_ds.assign_coords(longitude=new_ds.coords["longitude"] % 360.0) 


output_dir = os.path.dirname(args.output)
Path(output_dir).mkdir(parents=True, exist_ok=True)

new_ds.to_netcdf(args.output)
