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

sel_time = pd.Timestamp("2011-01-17")
test_range = 10
timedeltas = [ pd.Timedelta(days=t) for t in np.arange(- test_range, test_range+1) ]

sel_lat = np.arange(20, 60.5, 1)
sel_lon = np.arange(120, 240.5, 1)



varnames = ["mean_sea_level_pressure",]

ref_data = None 
    
S2Scoor = dict(L=0)



rms = np.zeros((len(varnames), len(timedeltas)))
maxdiff = rms.copy()

    
for i, varname in enumerate(varnames):
        
    print("Doing variable: ", varname)

    varname_ECCC = varname_ECCC_mapping[varname]
    varname_ERA5 = ERA5_loader.ERA5_longshortname_mapping[varname]
    ds = xr.open_dataset(url % (varname_ECCC,), decode_times=True).sel(S=sel_time).mean(dim="M").isel(**S2Scoor)
    dataA = ds[varname_ECCC].sel(X=sel_lon, Y=sel_lat)

    for j, timedelta in enumerate(timedeltas):
        print("Doing timedelta: ", timedelta)
        
        ref_data = ERA5_loader.readERA5(sel_time + timedelta, 24, varname)[varname_ERA5].isel(time=0).sel(longitude=sel_lon, latitude=sel_lat)

        maxdiff[i, j] = np.amax(np.abs(dataA.to_numpy() - ref_data.to_numpy()))
        rms[i, j]     = np.std(dataA.to_numpy() - ref_data.to_numpy())

        if i == 0 and j == 0:
            X = ref_data.coords["longitude"]
            Y = ref_data.coords["latitude"]

factor = 1e2

print("Loading matplotlib...")
import matplotlib.pyplot as plt
print("done.")

fig, ax = plt.subplots(len(varnames), 1, squeeze=False)
    
t = np.array( [ timedelta.total_seconds()/86400 for timedelta in timedeltas] )

for i, varname in enumerate(varnames):
    _ax = ax[i, 0]
    _ax.plot(t, rms[i, :], "b-" , label="RMS")
    _ax.plot(t, maxdiff[i, :], "b--", label="MaxDiff")
    _ax.set_title("%s" % (varname,))
    _ax.grid()
    _ax.legend()

fig.savefig("test.png", dpi=200)
plt.show()
