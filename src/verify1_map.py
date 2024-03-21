import xarray as xr
import numpy as np
import pandas as pd
import ECCC_tools
import ERA5_loader


target_dates = {
    'GEPS5': pd.Timestamp("2017-01-03"),
    'GEPS6': pd.Timestamp("2017-01-02"),
}

data = {}
varset = 'surf_inst'
long_varname = 'mean_sea_level_pressure'

ECCC_shortvarname = ECCC_tools.ECCC_longshortname_mapping[long_varname]
ERA5_shortvarname = ERA5_loader.ERA5_longshortname_mapping[long_varname]
number = 0

lat = None
lon = None
for model_version in ['GEPS5', "GEPS6"]:
    
    target_date = target_dates[model_version]

    print("Model version: %s, target_date: %s" % (
        model_version,
        target_date.strftime("%Y-%m-%d")
    ))


    
    data[model_version] = {
        'ECCC': ECCC_tools.open_dataset(varset, model_version, target_date)[ECCC_shortvarname].isel(start_time=0, lead_time=0).sel(number=number),
        'ERA5': ERA5_loader.readERA5(target_date + pd.Timedelta(days=1), dhr=24, varname=long_varname)[ERA5_shortvarname].isel(time=0),
    }

    # Check lat lon
    _data = data[model_version]

    print(_data['ECCC'])
    print(_data['ERA5'])

    diff_lat = _data['ECCC'].coords["latitude"].to_numpy()  - _data['ERA5'].coords['latitude'].to_numpy()
    diff_lon = _data['ECCC'].coords["longitude"].to_numpy() - _data['ERA5'].coords['longitude'].to_numpy()
    
    print('ECCC long: ', _data['ECCC'].coords["longitude"].to_numpy())
    print('ERA5 long: ', _data['ERA5'].coords['longitude'].to_numpy())

    print('ECCC lat: ', _data['ECCC'].coords["latitude"].to_numpy())
    print('ERA5 lat: ', _data['ERA5'].coords['latitude'].to_numpy())
    
    if np.mean(np.abs(diff_lat)) > 1e-5:
        raise Exception("Latitude is too different.")

    if np.mean(np.abs(diff_lon)) > 1e-5:
        raise Exception("Longitude is too different.")


    if lat is None:
        lat = diff_lon = _data['ECCC'].coords["latitude"].to_numpy()
        lon = diff_lon = _data['ECCC'].coords["longitude"].to_numpy()
    # print(diff_lon)




print("Loading matplotlib...")
import matplotlib as mplt
#mplt.use("TkAgg")
mplt.use("Agg")


import matplotlib.pyplot as plt
print("done.")

levs = np.arange(980, 1040, 5)
difflevs = np.linspace(-10, 10, 21)

fig, ax = plt.subplots(3, 2)

for i, model_version in enumerate(data.keys()):
    print("Plotting : ", model_version) 

    _ax = ax[:, i]
    _data = data[model_version]

    _ECCC = _data['ECCC'].to_numpy() / 1e2 # to hPa
    _ERA5 = _data['ERA5'].to_numpy() / 1e2
    _diff = _ECCC - _ERA5
 
    for j, __ax in enumerate(_ax[:2]):
        mappable = __ax.contourf(
            lon,
            lat,
            [_ECCC, _ERA5][j],
            levels = levs,
            cmap = 'jet',
        )
        plt.colorbar(mappable, ax = __ax, orientation="vertical")

    __ax = _ax[2]
    mappable = __ax.contourf(
        lon,
        lat,
        _diff,
        levels = difflevs,
        cmap = 'bwr',
    )
    plt.colorbar(mappable, ax = __ax, orientation="vertical")

    _ax[0].set_title("%s, %s" % (model_version, target_dates[model_version].strftime("%Y-%m-%d")))

fig.savefig("fig_verify1_map.png", dpi=200)
plt.show()
