import xarray as xr
import numpy as np
import pandas as pd
import traceback
import os

test_varname = "psl"

dataset_infos = dict(
    
    ECCC_GEPS5 = dict(
        url = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS5/.{forecast_type:s}/{varname:s}/dods",
    ),

    ECCC_GEPS6 = dict(
        url = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS6/.{forecast_type:s}/{varname:s}/dods",
    ),
)


data = dict()

print("Test varname: ", test_varname)

for forecast_type in ["hindcast", "forecast"]:
    
    subdata = dict() 
    for model in ["ECCC_GEPS5", "ECCC_GEPS6"]:
       
        print("# Model: %s" % (model,))
        output_filename = "detecting-dataset_%s_%s.nc" % (forecast_type, model,)
        print("Output filename: ", output_filename)

        if os.path.exists(output_filename):
            print("File already exists. Simply load it.")
            subdata[model] = xr.open_dataset(output_filename)
            continue


        dataset_info = dataset_infos[model]
        
        url_fmt = dataset_info["url"]
        ds = xr.open_dataset(
            url_fmt.format(
                forecast_type=forecast_type,
                varname=test_varname
        ), decode_times=True)

        dims = ds.dims
        da = ds[test_varname].isel(X=0, Y=0, L=0, M=0)

        da = da.where(np.isfinite(da), drop=True)
        
        ts = da.coords["S"].rename("time")
        
        print("Writing output file:", output_filename)
        ts.to_netcdf(output_filename)

        subdata[model] = xr.open_dataset(output_filename)

    data[forecast_type] = subdata

print("Done loading files.")
print("Now figure out stuff... ")
print("# Finding common time")

for forecast_type in ["hindcast", "forecast"]:
   
    print("## Simulation type: ", forecast_type)

    common_times = np.intersect1d(
        data[forecast_type]["ECCC_GEPS5"]["time"],
        data[forecast_type]["ECCC_GEPS6"]["time"],
    )

    print("Found common times: %d" % (len(common_times)))
    for i, common_time in enumerate(common_times):
        
        print("[%d] %s" % (i, str(common_time)))
        






