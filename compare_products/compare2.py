import xarray as xr
import numpy as np
import pandas as pd
import ERA5_loader

dataset_infos = dict(

    ECCC_GEPS5 = dict(
        url = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS5/.hindcast/%s/dods",
        valid_date = "2011-01-03",
    ),

    ECCC_GEPS6 = dict(
        url = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS5/.hindcast/%s/dods",
        valid_date = "2011-01-07",
    ),

)



datasets = list(dataset_infos.keys())

data_interval = pd.Timedelta(days=7)

varname_ECCC_mapping = {
    "mean_sea_level_pressure" : "psl",
}

start_time_range = [
    pd.Timestamp("2011-01-03"),
    pd.Timestamp("2011-01-03"),
]


sel_time = pd.Timestamp("2011-01-17")
test_range = 3
timedeltas = [ pd.Timedelta(days=t) for t in np.arange(test_range) ]
hr12 = pd.Timedelta(hours=12)


sel_lat = np.arange(20, 60.5, 1)
sel_lon = np.arange(120, 240.5, 1)



varnames = ["mean_sea_level_pressure",]

ref_data = None 
    
S2Scoor = dict(L=0)



rms = np.zeros((len(datasets), len(varnames), len(timedeltas)))
maxdiff = rms.copy()


for i, dataset in enumerate(datasets):
        
    print("Doing dataset: ", dataset)

    dataset_info = dataset_infos[dataset]
    url = dataset_info["url"]    
 
    for j, varname in enumerate(varnames):
            
        print("Doing variable: ", varname)

        varname_ECCC = varname_ECCC_mapping[varname]
        varname_ERA5 = ERA5_loader.ERA5_longshortname_mapping[varname]

        for k, timedelta in enumerate(timedeltas):
            print("Doing timedelta: ", timedelta)
     
            ds = xr.open_dataset(url % (varname_ECCC,), decode_times=True).sel(S=sel_time).mean(dim="M").sel(L=timedelta + hr12)
          
            dataA = ds[varname_ECCC].sel(X=sel_lon, Y=sel_lat)
           
            ref_data = ERA5_loader.readERA5(sel_time + timedelta, 24, varname)[varname_ERA5].isel(time=0).sel(longitude=sel_lon, latitude=sel_lat)

            maxdiff[i, j, k] = np.amax(np.abs(dataA.to_numpy() - ref_data.to_numpy()))
            rms[i, j, k]     = np.std(dataA.to_numpy() - ref_data.to_numpy())

            if i == 0 and j == 0 and k == 0:
                X = ref_data.coords["longitude"]
                Y = ref_data.coords["latitude"]

factor = 1e2

print("Loading matplotlib...")
import matplotlib.pyplot as plt
print("done.")

fig, ax = plt.subplots(len(varnames), 1, squeeze=False)
    
t = np.array( [ timedelta.total_seconds()/86400 for timedelta in timedeltas] )

    
for j, varname in enumerate(varnames):
    _ax = ax[j, 0]
    for i, dataset in enumerate(datasets):
        _ax.plot(t, rms[i, j, :], label="%s" % (dataset,))
        #_ax.plot(t, maxdiff[i, i, :], "b--", label="MaxDiff")
        _ax.set_title("%s" % (varname,))
        _ax.grid()
        _ax.legend()

fig.savefig("test.png", dpi=200)
plt.show()
