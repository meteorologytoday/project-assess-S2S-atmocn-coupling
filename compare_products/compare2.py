import xarray as xr
import numpy as np
import pandas as pd
import ERA5_loader

S2S_dataset = "ECCC_GEPS5"

url = dict(
    ECCC_GEPS5 = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS5/.hindcast/%s/dods",
    ECCC_GEPS6 = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS6/.hindcast/%s/dods",
)[S2S_dataset]

varname_ECCC_mapping = {
    "mean_sea_level_pressure" : "psl",
}

data = {
    S2S_dataset : {},
    "ERA5" : {},
}

sel_time = pd.Timestamp("2011-01-03")


sel_lat = np.arange(20, 60.5, 1)
sel_lon = np.arange(120, 240.5, 1)



varnames = ["mean_sea_level_pressure",]

ref_data = None 
    
S2Scoor = dict(L=0, M=2)

for varname in varnames:

    varname_ECCC = varname_ECCC_mapping[varname]
    varname_ERA5 = ERA5_loader.ERA5_longshortname_mapping[varname]

    ds = xr.open_dataset(url % (varname_ECCC,), decode_times=True).sel(S=sel_time).isel(**S2Scoor)

    data[S2S_dataset][varname] = ds[varname_ECCC].sel(X=sel_lon, Y=sel_lat)

    data["ERA5"][varname] = ERA5_loader.readERA5(sel_time, 24, varname)[varname_ERA5].isel(time=0).sel(longitude=sel_lon, latitude=sel_lat)

    if ref_data is None:
        ref_data = data["ERA5"][varname]
        X = ref_data.coords["longitude"]
        Y = ref_data.coords["latitude"]

factor = 1e2

print("Loading matplotlib...")
import matplotlib.pyplot as plt
print("done.")

fig, ax = plt.subplots(3, 1)

levs = np.arange(980, 1040, 5)
levs_diff = np.linspace(-1, 1, 21) * 10

dataA = data["ERA5"][varname].to_numpy() / factor
dataB = data[S2S_dataset][varname].to_numpy() / factor

mappable      = ax[0].contourf(X, Y, dataA, levs, cmap="jet")
mappable      = ax[1].contourf(X, Y, dataB, levs, cmap="jet")
mappable_diff = ax[2].contourf(X, Y, dataA - dataB, levs_diff, cmap="bwr")

cb = plt.colorbar(mappable, ax=ax[0], orientation="vertical")
cb = plt.colorbar(mappable, ax=ax[1], orientation="vertical")
cb_diff = plt.colorbar(mappable_diff, ax=ax[2], orientation="vertical")

ax[0].set_title("ERA5: %s" % (sel_time.strftime("%Y-%m-%d %H"),) )
ax[1].set_title("%s: %s" % (S2S_dataset, sel_time.strftime("%Y-%m-%d %H"), ))

fig.savefig("test.png", dpi=200)
plt.show()
