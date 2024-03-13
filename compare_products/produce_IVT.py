import xarray as xr
import numpy as np
import pandas as pd



urls = dict(
    GEPS5 = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS5/.hindcast/%s/dods",
    GEPS6 = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS6/.hindcast/%s/dods",
)

dataset_names = ["GEPS5", "GEPS6"]
data = dict()

# huss = specific humidity
# ua   = zonal wind
# va   = meridional wind

sel_dict = dict(
    S = pd.Timestamp("1999-01-01"),
    L = pd.Timedelta(hours=12, days=0),
)

isel_dict = dict(M=0)
ref_data = None
for dataset_name in dataset_names:

    data[dataset_name] = dict()
    url_fmt = urls[dataset_name]
    
    for varname in ["huss", "ua", "va"]:
        data[dataset_name] = xr.open_dataset(url % (varname,), decode_times=True).sel(**sel_dict).isel(**isel_dict)
        
        if ref_data is None:
            ref_data = data[dataset_name][varname]
            X = ref_data.coords["X"]
            Y = ref_data.coords["Y"]

    # Compute IWV IVT
     


print("Loading matplotlib...")
import matplotlib as mplt
mplt.use("TkAgg")


import matplotlib.pyplot as plt
print("done.")

fig, ax = plt.subplots(1, 1)


im = ax.imshow(rms)

cbar = plt.colorbar(im, ax=ax)

ax.set_ylabel("Start Time")
ax.set_xlabel("Lead Time")

fig.savefig("lead_start_verify.png", dpi=200)
plt.show()
