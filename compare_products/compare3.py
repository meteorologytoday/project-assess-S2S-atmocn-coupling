import numpy as np
import netCDF4
import xarray as xr
import os
import pandas as pd
import scipy
import ERA5_loader









if __name__ == "__main__":

    dt = pd.Timestamp("2011-01-02")
    dhr = 24
    varname = "surface_pressure"
    ds = readERA5(dt, dhr, varname)

    print(ds)
   
    print("Data is read.") 
    print("Load matplotlib...")
    import matplotlib as mplt
    mplt.use("TkAgg")

    import matplotlib.pyplot as plt
    print("Done")

    fig, ax = plt.subplots(1, 1)
    
    mappable = ax.contourf(ds.coords["longitude"], ds.coords["latitude"], ds[ERA5_longshortname_mapping[varname]].isel(time=0))
    plt.colorbar(mappable, ax = ax, orientation="vertical")
    
    plt.show()
