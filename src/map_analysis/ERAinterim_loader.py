import numpy as np
import netCDF4
import xarray as xr
import os
import pandas as pd
import scipy

archive_root = "ERAinterim"

# Find the first value that is True
def findfirst(xs):

    for i, x in enumerate(xs):
        if x:
            return i

    return -1

# Read ERA5 data and by default convert to S2S project
# resolution (1deg x 1deg), and rearrange the axis in
# a positively increasing order.
def open_dataset_ERAinterim(dt_rng, dhr, varset, varname, inclusive="left"):
   
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
        
        filename = "ERAInterim-%s.nc" % (
            dt.strftime("%Y-%m-%d_%H"),
        )

        full_filename = os.path.join(archive_root, varset, "%02dhr" % (dhr,), filename)

        filenames.append(full_filename)

    ds = xr.open_mfdataset(filenames)

    if varname in ["IVT", "IVT_x", "IVT_y", "IWV"]:
        #print("latitude: ", ds["lat"])
        #print("longitude: ", ds["lon"])

        lat = ds["lat"].to_numpy()
        lon = ds["lon"].to_numpy()
        ds = ds.drop_vars(["lat", "lon"]).assign_coords(
            dict(
                latitude=lat,
                longitude=lon,
            )
        )


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

 
    # Finally flip latitude
    lat = ds.coords["latitude"].to_numpy()
    if np.all( (lat[1:] - lat[:-1]) < 0 ):
        print("Flip latitude so that it is monotonically increasing")
        ds = ds.isel(latitude=slice(None, None, -1))
    
    return ds



if __name__ == "__main__":

    dt = pd.Timestamp("2011-01-02")
    dhr = 24
    varset  = "sfc"
    varname = "msl"
    ds = open_dataset_ERAinterim(dt, dhr, varset, varname)

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
