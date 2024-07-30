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

ocean_regions = [
    "ALL", 
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
base_ocean_mask = mask.to_numpy().astype(int)

# Copy is necessary so that the value can be assigned later
mask = mask.expand_dims(
    dim = dict(ocean_region=ocean_regions),
    axis=0,
).rename("ocean_mask").copy()

ocean_masks = np.zeros(
    (len(ocean_regions), len(mask.coords["latitude"]), len(mask.coords["longitude"]),),
    dtype=int,
)

for i, ocean_region in enumerate(ocean_regions):
   
    print("Making region: ", ocean_region) 
    _mask = mask.sel(ocean_region=ocean_region).copy()
    
    if ocean_region == "ARC":
        
        _mask = _mask.where(
            (_mask.latitude > 60) 
        )
 
    elif ocean_region == "N-PAC":
        
        _mask = _mask.where(
            (_mask.latitude > 30) &
            (_mask.latitude <= 60) &
            (_mask.longitude > 120) &
            (_mask.longitude <= 250) 
        )
    
    elif ocean_region == "T-PAC":
        
        _mask = _mask.where(
            (_mask.latitude <= 30) &
            (_mask.latitude > -30) &
            (_mask.longitude > 120) &
            (_mask.longitude <= 260) 
        )
 
    elif ocean_region == "S-PAC":
        
        _mask = _mask.where(
            (_mask.latitude <= -30) &
            (_mask.latitude > -60)  &
            (_mask.longitude > 120) &
            (_mask.longitude <= 290) 
        )
 
    elif ocean_region == "T-IND":
        
        _mask = _mask.where(
            (_mask.latitude <= 30) &
            (_mask.latitude > -30)  &
            (_mask.longitude > 25) &
            (_mask.longitude <= 120) 
        )
 
    elif ocean_region == "S-IND":
        
        _mask = _mask.where(
            (_mask.latitude <= -30) &
            (_mask.latitude > -60)  &
            (_mask.longitude > 25) &
            (_mask.longitude <= 120) 
        )
    elif ocean_region == "N-ATL":
        
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
 
    elif ocean_region == "T-ATL":
        
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
 
    elif ocean_region == "S-ATL":
        
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
 
    elif ocean_region == "SO":
        
        _mask = _mask.where(
            (_mask.latitude <= -60)
        )
 
 
    _mask = xr.apply_ufunc(np.isfinite, _mask)
    mask[i, :, :] = _mask.to_numpy().astype(int) * base_ocean_mask

#mask[:, :, :] = ocean_masks

new_ds = xr.merge([mask, ])
new_ds.to_netcdf(args.output)






