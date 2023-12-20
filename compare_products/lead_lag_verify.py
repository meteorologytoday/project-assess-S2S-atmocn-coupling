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

coor = dict(M=0)

start_times = pd.date_range(start="1999-01-01", end="1999-01-10", freq="D")
lead_times = list(range(5))


rms = np.zeros((len(start_times), len(lead_times)))


for i, start_time in enumerate(start_times):
    for j, lead_time in enumerate(lead_times):

        print("Loading data of (i, j) = (%d, %d)" % (i, j,))

        S = start_time
        L = pd.Timedelta(hours=12, days=lead_time)


        dataA = data[0][varname].isel(**coor).sel(S=S, L=pd.Timedelta(hours=12))
        dataB = data[0][varname].isel(**coor).sel(S=S, L=L)
        
        rms[i, j] = np.std(dataA - dataB)


print("Loading matplotlib...")
import matplotlib.pyplot as plt
print("done.")

fig, ax = plt.subplots(3, 1)


im = ax.imshow(rms)

cbar = plt.colorbar(im, ax=ax)


plt.show()
