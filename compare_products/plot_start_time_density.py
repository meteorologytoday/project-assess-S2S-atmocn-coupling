import numpy as np
import xarray as xr
import pandas as pd

beg_year = 1998
end_year = 2018
total_years = end_year - beg_year


time_files = dict(
    GEPS5 = "detecting-dataset_hindcast_ECCC_GEPS5.nc",
    GEPS6 = "detecting-dataset_hindcast_ECCC_GEPS6.nc",
)

datasets = list(time_files.keys())
data = dict()
for dataset, time_file in time_files.items():
    data[dataset] = xr.open_dataset(time_file)

cnt = dict()
for dataset, _ in time_files.items():
    _cnt = np.zeros((total_years, 12,))
    month_vec = data[dataset]["time"].dt.month.to_numpy()
    year_vec  = data[dataset]["time"].dt.year.to_numpy()

    for y, year in enumerate(range(beg_year, end_year)):
        for m, month in enumerate(range(1, 13)):
            _cnt[y, m] = np.sum((month_vec==month) & (year_vec == year))
            #print("%04d-%02d : %d" % (year, month, _cnt[y,m]))

    cnt[dataset] = _cnt 


dts = pd.date_range(
    start="%04d-01-01" % (beg_year,),
    end="%04d-01-01" % (end_year,),
    freq="MS",
    inclusive="left",
)


print("Loading matplotlib...")
import matplotlib as mplt
import matplotlib.pyplot as plt
print("done.")


fig, ax = plt.subplots(len(datasets), 1, figsize=(8, 8), squeeze=False)

time_vec = beg_year + np.arange(len(dts)) / 12

for i, _ax in enumerate(ax.flatten()):
    dataset = datasets[i]
    _ax.bar(time_vec, cnt[dataset].flatten())
    #_ax.set_xticks(range(12), ["%02d" % (m+1,) for m in range(12)])
    _ax.set_title("Dataset: %s" % (dataset,))
    _ax.set_ylabel("Monthly Count")

    _ax.grid()


fig.savefig("fig_all_months_density.png", dpi=200)

 
fig2, ax = plt.subplots(len(datasets), 1, figsize=(8, 8), squeeze=False)
time_vec = np.arange(12) + 1

for i, _ax in enumerate(ax.flatten()):
    
    dataset = datasets[i]
    _ax.bar(time_vec, cnt[dataset].sum(axis=0))
    _ax.set_title("Dataset: %s" % (dataset,))
    _ax.set_ylabel("Monthly Count")
    _ax.set_xticks(time_vec) 
    _ax.grid()

fig2.savefig("fig_monthly_density.png", dpi=200)

 
plt.show()





