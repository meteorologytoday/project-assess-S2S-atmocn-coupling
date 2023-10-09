import numpy as np
import netCDF4
import xarray as xr
import os
import pandas as pd
import scipy

archive_root = "data/ERA5"

ERA5_longshortname_mapping = dict(
    surface_pressure = "sp",
    mean_sea_level_pressure = "msl",
)

# Find the first value that is True
def findfirst(xs):

    for i, x in enumerate(xs):
        if x:
            return i

    return -1

# Read ERA5 data and by default convert to S2S project
# resolution (1deg x 1deg), and rearrange the axis in
# a positively increasing order.
def readERA5(dt_rng, dhr, varname, if_downscale = True, inclusive="left"):
   
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
            varname,
            dt.strftime("%Y-%m-%d_%H"),
        )

        full_filename = os.path.join(archive_root, "%02dhr" % (dhr,), varname, filename)

        filenames.append(full_filename)

    ds = xr.open_mfdataset(filenames)
    ds = ds.isel(latitude=slice(None, None, -1))

    lat = ds.coords["latitude"].to_numpy()
    lon = ds.coords["longitude"].to_numpy()

    #first_decrease_idx = findfirst( (lon[1:] - lon[:-1]) < 0 )
    #if first_decrease_idx != -1:
    #    roll_by = - (first_decrease_idx + 1)
    #    ds = ds.roll(longitude=roll_by)

    first_positive_idx = findfirst( lon > 0 )
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


    if if_downscale:
        
        short_varname = ERA5_longshortname_mapping[varname]
        dims = len(ds[short_varname].shape)
        axes = (dims-2, dims-1)
        
        # We are using the fact that ERA5 resultion is 0.25
        # so the size is picked 5
        ds[short_varname][:] = scipy.ndimage.uniform_filter(
            ds[short_varname], size=5, axes=axes,
        )

        lon = ds.coords["longitude"].to_numpy()
        lat = ds.coords["latitude"].to_numpy()
        first_lon_integer_idx = findfirst( lon % 1.0 == 0.0 )
        first_lat_integer_idx = findfirst( lat % 1.0 == 0.0 )


        if first_lat_integer_idx == -1:
            raise Exception("Cannot find latitude in integer")

        if first_lon_integer_idx == -1:
            raise Exception("Cannot find longitude in integer")
      
        ds = ds.isel(
            latitude  = slice(first_lat_integer_idx, None, 4),
            longitude = slice(first_lon_integer_idx, None, 4),
        )
        

    return ds



if __name__ == "__main__":

    dt = pd.Timestamp("2011-01-02")
    dhr = 24
    varname = "surface_pressure"
    ds = readERA5(dt, dhr, varname)

    print(ds)
   
    print("Data is read.") 
    print("Load matplotlib...")
    import matplotlib.pyplot as plt
    print("Done")

    fig, ax = plt.subplots(1, 1)
    
    mappable = ax.contourf(ds.coords["longitude"], ds.coords["latitude"], ds[ERA5_longshortname_mapping[varname]].isel(time=0))
    plt.colorbar(mappable, ax = ax, orientation="vertical")
    
    plt.show()
