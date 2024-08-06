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

parser.add_argument('--test-date', type=str, default="1998-01-03")
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()
print(args)

test_dt = pd.Timestamp(args.test_date)


dlon = 5
dlat = 5

scan_lon_rng = [360-160, 360-115]
scan_lat_rng = [30, 65]

nregions = int( (scan_lon_rng[1] - scan_lon_rng[0]) / dlon )
regions = ["%04d" % (i+1,) for i in range(nregions)]



model_version = "GEPS5"
model_version_date = ECCC_tools.modelVersionReforecastDateToModelVersionDate(model_version, test_dt)


if model_version_date is None:
    raise Exception("Wrong test_dt %s" % (str(test_dt),))

ds = ECCC_tools.open_dataset("raw", "surf_avg", model_version, test_dt) 

test_da = ds["sst"].isel(start_time=0, lead_time=0, number=0)

full_mask = xr.ones_like(test_da).astype(int)
lnd_mask = xr.apply_ufunc(np.isnan, test_da).astype(int)

# Copy is necessary so that the value can be assigned later
mask = xr.zeros_like(full_mask).expand_dims(
    dim = dict(region=regions),
    axis=0,
).rename("mask").copy()

regions = []

for lon_beg in np.arange(scan_lon_rng[0], scan_lon_rng[1], dlon):
    for lat_beg in np.arange(scan_lat_rng[0], scan_lat_rng[1], dlat):
        
        lon_end = lon_beg + dlon
        lat_end = lat_beg + dlat
        
        # test if hit land.
        test_idx = (
            (lnd_mask.coords["latitude"]  >   lat_beg) &
            (lnd_mask.coords["latitude"]  <=  lat_end) &
            (lnd_mask.coords["longitude"] >   lon_beg) &
            (lnd_mask.coords["longitude"] <=  lon_end) 

        )
        
        cnt_land = lnd_mask.where(test_idx).sum(dim=["longitude", "latitude"], skipna=True).to_numpy()
        print(lnd_mask.where(test_idx))
        print(cnt_land)
        
        if cnt_land == 0:
            continue
        else:
            i = len(regions)
            regions.append("%04d" % (i+1,))
        
            _mask = full_mask.copy().where(test_idx)
            _mask = xr.apply_ufunc(np.isfinite, _mask)
            mask[i, :, :] = _mask.to_numpy().astype(int)
            break
        
new_ds = xr.merge([mask, ])
new_ds.to_netcdf(args.output)






