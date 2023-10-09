import xarray as xr
import numpy as np
import pandas as pd
urls = [
    "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS5/.hindcast/%s/dods",
    "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS6/.hindcast/%s/dods",
]

data = []
varname = "psl"
for i, url in enumerate(urls):
    data.append(xr.open_dataset(url % (varname,), decode_times=True))



ref_data = data[0]

X = ref_data.coords["X"]
Y = ref_data.coords["Y"]

print(X)
print(Y)

coor = dict(L=0, M=0)
sel_timeA  = pd.Timestamp("1998-01-03")
sel_timeA2 = pd.Timestamp("1998-01-03")
sel_timeB = pd.Timestamp("1998-01-07")
factor = 1e2
dataA = data[0][varname].isel(**coor).sel(S=sel_timeA) / factor
dataA2 = data[0][varname].isel(**coor).sel(S=sel_timeA2) / factor

dataA = (dataA + dataA2) / 2

dataB = data[1][varname].isel(**coor).sel(S=sel_timeB) / factor


print("Loading matplotlib...")
import matplotlib.pyplot as plt
print("done.")

fig, ax = plt.subplots(3, 1)

levs = np.arange(980, 1040, 5)
levs_diff = np.linspace(-1, 1, 21) * 10

mappable      = ax[0].contourf(X, Y, dataA, levs, cmap="jet")
mappable      = ax[1].contourf(X, Y, dataB, levs, cmap="jet")
mappable_diff = ax[2].contourf(X, Y, dataA - dataB, levs_diff, cmap="bwr")

cb = plt.colorbar(mappable, ax=ax[0], orientation="vertical")
cb = plt.colorbar(mappable, ax=ax[1], orientation="vertical")
cb_diff = plt.colorbar(mappable_diff, ax=ax[2], orientation="vertical")

ax[0].set_title(sel_timeA.strftime("%Y-%m-%d %H"))
ax[1].set_title(sel_timeB.strftime("%Y-%m-%d %H"))

plt.show()
