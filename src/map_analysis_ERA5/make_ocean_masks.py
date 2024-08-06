from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import argparse
import traceback
import os
import pretty_latlon
pretty_latlon.default_fmt = "%d"

import ECCC_tools
ECCC_tools.archive_root = os.path.join("S2S", "ECCC", "data20_20240723")


import ERA5_loader

parser = argparse.ArgumentParser(
                    prog = 'make_ECCC_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

#parser.add_argument('--start-months', type=int, nargs="+", required=True)
parser.add_argument('--test-date', type=str, default="1998-01-03")
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()
print(args)

test_dt = pd.Timestamp(args.test_date)

regions = [
    "ALL", 
    "NW-PAC", "NE-PAC",
    "N-PAC", "T-PAC", "S-PAC",
    "T-IND", "S-IND",
    "N-ATL", "T-ATL", "S-ATL",
    "ARC", "SO",
]



model_version = "GEPS5"
model_version_date = ECCC_tools.modelVersionReforecastDateToModelVersionDate(model_version, test_dt)


if model_version_date is None:
    raise Exception("Wrong test_dt %s" % (str(test_dt),))

ds = ECCC_tools.open_dataset("raw", "surf_avg", model_version, test_dt) 

test_da = ds["sst"].isel(start_time=0, lead_time=0, number=0)

mask = xr.apply_ufunc(np.isfinite, test_da)
base_mask = mask.to_numpy().astype(int)

# Copy is necessary so that the value can be assigned later
mask = mask.expand_dims(
    dim = dict(region=regions),
    axis=0,
).rename("mask").copy()

masks = np.zeros(
    (len(regions), len(mask.coords["latitude"]), len(mask.coords["longitude"]),),
    dtype=int,
)

for i, region in enumerate(regions):
   
    print("Making region: ", region) 
    _mask = mask.sel(region=region).copy()
    
    if region == "ARC":
        
        _mask = _mask.where(
            (_mask.latitude > 60) 
        )
 
    elif region == "N-PAC":
        
        _mask = _mask.where(
            (_mask.latitude > 30) &
            (_mask.latitude <= 60) &
            (_mask.longitude > 120) &
            (_mask.longitude <= 250) 
        )
 
    elif region == "NE-PAC":
        
        _mask = _mask.where(
            (_mask.latitude > 30) &
            (_mask.latitude <= 60) &
            (_mask.longitude > 180) &
            (_mask.longitude <= 250) 
        )
 
    elif region == "NW-PAC":
        
        _mask = _mask.where(
            (_mask.latitude > 30) &
            (_mask.latitude <= 60) &
            (_mask.longitude > 120) &
            (_mask.longitude <= 180) 
        )
    
    elif region == "T-PAC":
        
        _mask = _mask.where(
            (_mask.latitude <= 30) &
            (_mask.latitude > -30) &
            (_mask.longitude > 120) &
            (_mask.longitude <= 260) 
        )
 
    elif region == "S-PAC":
        
        _mask = _mask.where(
            (_mask.latitude <= -30) &
            (_mask.latitude > -60)  &
            (_mask.longitude > 120) &
            (_mask.longitude <= 290) 
        )
 
    elif region == "T-IND":
        
        _mask = _mask.where(
            (_mask.latitude <= 30) &
            (_mask.latitude > -30)  &
            (_mask.longitude > 25) &
            (_mask.longitude <= 120) 
        )
 
    elif region == "S-IND":
        
        _mask = _mask.where(
            (_mask.latitude <= -30) &
            (_mask.latitude > -60)  &
            (_mask.longitude > 25) &
            (_mask.longitude <= 120) 
        )
    elif region == "N-ATL":
        
        _mask = _mask.where(
            (
                (_mask.latitude <= 60) &
                (_mask.latitude >  30)  &
                (_mask.longitude > 260)
            ) 

            |
 
            (
                (_mask.latitude <= 60) &
                (_mask.latitude >  30)  &
                (_mask.longitude <= 10)
            ) 
        )
 
    elif region == "T-ATL":
        
        _mask = _mask.where(
            (
                (_mask.latitude <=  30) &
                (_mask.latitude >  -30)  &
                ( 
                    (_mask.longitude > 292) | 
                    (_mask.longitude <= 20)
                ) 
            ) 
        )
 
    elif region == "S-ATL":
        
        _mask = _mask.where(
            (
                (_mask.latitude <= -30) &
                (_mask.latitude >  -60)  &
                (
                    (_mask.longitude > 292) |
                    (_mask.longitude <= 20)
                ) 

            ) 
        )
 
    elif region == "SO":
        
        _mask = _mask.where(
            (_mask.latitude <= -60)
        )
 
 
    _mask = xr.apply_ufunc(np.isfinite, _mask)
    mask[i, :, :] = _mask.to_numpy().astype(int) * base_mask

#mask[:, :, :] = masks

new_ds = xr.merge([mask, ])
new_ds.to_netcdf(args.output)






