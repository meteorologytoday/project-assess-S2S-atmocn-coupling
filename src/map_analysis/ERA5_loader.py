import numpy as np
import netCDF4
import xarray as xr
import os
import pandas as pd
import scipy

archive_root = os.path.join("ERA5_global")

ERA5_longshortname_mapping = {
    "surface_pressure": "sp",
    "mean_sea_level_pressure": "msl",
    "10m_u_component_of_wind":  "u10",
    "10m_v_component_of_wind":  "v10",
}

ERA5_shortlongname_mapping = {
    shortname : longname
    for longname, shortname in ERA5_longshortname_mapping.items()
}

# Find the first value that is True
def findfirst(xs):

    for i, x in enumerate(xs):
        if x:
            return i

    return -1

# Read ERA5 data and by default convert to S2S project
# resolution (1deg x 1deg), and rearrange the axis in
# a positively increasing order.
def open_dataset_ERA5(dt_rng, dhr, shortname, if_downscale = True, inclusive="left"):
   
    longname = ERA5_shortlongname_mapping[shortname]
    
    if not hasattr(dt_rng, '__iter__'):
        dt_rng = [dt_rng, dt_rng]
        inclusive = "both"
    
    dts = pd.date_range(
        start=dt_rng[0],
        end=dt_rng[1],
        freq=pd.Timedelta(hours=dhr),
        inclusive=inclusive,
    )

    filenames = []
    for dt in dts:
        
        filename = "ERA5-%s-%s.nc" % (
            longname,
            dt.strftime("%Y-%m-%d_%H"),
        )

        full_filename = os.path.join(archive_root, "%02dhr" % (dhr,), longname, filename)

        filenames.append(full_filename)

    ds = xr.open_mfdataset(filenames)

    # flip latitude
    ds = ds.isel(latitude=slice(None, None, -1))

    lat = ds.coords["latitude"].to_numpy()
    lon = ds.coords["longitude"].to_numpy()


    # make longitude 0 the first element
    first_positive_idx = findfirst( lon >= 0 )
    if first_positive_idx != -1:
        roll_by = - first_positive_idx
        ds = ds.roll(longitude=roll_by).assign_coords(
            coords = {
                "longitude" : np.roll(
                    ds.coords["longitude"].to_numpy() % 360,
                    roll_by,
                )
            }
        )

    return ds



if __name__ == "__main__":

    dt = pd.Timestamp("2011-01-02")
    dhr = 24
    varname = "msl"
    ds = open_dataset_ERA5(dt, dhr, varname)

    print(ds)
   
    print("Data is read.") 
    print("Load matplotlib...")
    import matplotlib as mplt
    mplt.use("TkAgg")

    import matplotlib.pyplot as plt
    print("Done")

    fig, ax = plt.subplots(1, 1)
    
    mappable = ax.contourf(ds.coords["longitude"], ds.coords["latitude"], ds[varname].isel(time=0))
    plt.colorbar(mappable, ax = ax, orientation="vertical")
    
    plt.show()
