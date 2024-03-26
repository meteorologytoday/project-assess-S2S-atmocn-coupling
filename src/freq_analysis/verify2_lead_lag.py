import xarray as xr
import numpy as np
import pandas as pd
import ECCC_tools
import ERA5_loader


target_dates = {
    'GEPS5': pd.Timestamp("2005-02-28"),
    'GEPS6': pd.Timestamp("2005-02-27"),
}

prediction_step = 0

timedeltas = [
    pd.Timedelta(days=i) for i in np.arange(-15, 16)
]

timedeltas_np = [
    timedeltas[i] / pd.Timedelta(days=1)
    for i in range(len(timedeltas))    
]

print(timedeltas_np)

data = {}
varset = 'surf_inst'
long_varname = 'mean_sea_level_pressure'

ECCC_shortvarname = ECCC_tools.ECCC_longshortname_mapping[long_varname]
ERA5_shortvarname = ERA5_loader.ERA5_longshortname_mapping[long_varname]
number = 0

lat = None
lon = None

model_versions = ['GEPS5', "GEPS6"]

data = {}

for model_version in model_versions:
    
    target_date = target_dates[model_version]

    print("Model version: %s, target_date: %s" % (
        model_version,
        target_date.strftime("%Y-%m-%d")
    ))


        
    ECCC_data = ECCC_tools.open_dataset("raw", varset, model_version, target_date)[ECCC_shortvarname].isel(start_time=0, lead_time=prediction_step).sel(number=number)
    
    res = np.zeros((len(timedeltas), ))
    for i, timedelta in enumerate(timedeltas):
        ERA5_date = target_date + timedelta
        print("Loading timedelta = ", timedelta, "; date = ", ERA5_date)
        ERA5_data = ERA5_loader.readERA5(ERA5_date, dhr=24, varname=long_varname)[ERA5_shortvarname].isel(time=0)

        diff = ECCC_data.to_numpy() - ERA5_data.to_numpy()

        res[i] = np.sqrt(np.mean(diff**2)) / 1e2 # hPa

    data[model_version] = {
        'res' : res,
    }

print("Loading matplotlib...")
import matplotlib as mplt
mplt.use("Agg")


import matplotlib.pyplot as plt
print("done.")


fig, ax = plt.subplots(1, 2)

for i, model_version in enumerate(model_versions):
    
    print("Plotting : ", model_version) 

    _ax = ax[i]


    _ax.plot(timedeltas_np, data[model_version]['res'])
    _ax.set_title("%s, %s" % (model_version, target_dates[model_version].strftime("%Y-%m-%d")))


    _ax.grid()
    _ax.set_xlabel("Offset from target date [day]")
    _ax.set_ylabel("Mean residue [hPa]")
    _ax.set_ylim([0, 20])

fig.savefig("fig_verify2_lead_lag.png", dpi=200)
plt.show()
