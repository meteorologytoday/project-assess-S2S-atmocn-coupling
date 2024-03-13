import xarray as xr
import numpy as np
import scipy
import pandas as pd
import argparse
import os.path
import time_tools


def findLargeWindow(dts, window_secs = 7*86400.0):
    
    delta_dts = dts[1:] - dts[:-1]
    large_windows = []

    for i, delta_dt in enumerate(delta_dts):
        
        delta_secs = delta_dt / np.timedelta64(1, 's')

        if delta_secs > window_secs:
            
            large_windows.append((dts[i], dts[i+1]))


    return large_windows



 
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-dirs', type=str, nargs="+", help='Input directory.', required=True)
parser.add_argument('--year-rng', type=str, nargs=2, help='Range of time. Format: yyyy-mm', required=True)
parser.add_argument('--output', type=str, help='Output filename in png.', default="")
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

varname = "mean_sea_level_pressure_RMS"

beg_ym_parsed = args.year_rng[0].split("-")
end_ym_parsed = args.year_rng[1].split("-")

beg_ym = pd.Timestamp(year=int(beg_ym_parsed[0]), month=int(beg_ym_parsed[1]), day=1)
end_ym = pd.Timestamp(year=int(end_ym_parsed[0]), month=int(end_ym_parsed[1]), day=1)

dts = pd.date_range(start=beg_ym, end=end_ym, freq="MS")


data = []
ref_time = pd.Timestamp("1970-01-01 00:00:00")

# 5 days per week
fcst_weeks = np.array([0, 1])


for input_dir in args.input_dirs:
    filenames = []
    for dt in dts:
        filenames.append(
            os.path.join(input_dir, "fcst_error_%s.nc" % (dt.strftime("%Y-%m"),))
        )
  
    print(filenames) 
    ds = xr.open_mfdataset(filenames, concat_dim="start_time", combine="nested")


    # Coarsen the data
    _ds = ds.coarsen({"lead_time":5}, boundary="trim").mean().rename({"lead_time": "lead_week"})

    data.append(_ds)


for i, _data in enumerate(data):

    print("Interpolating dataset %d" % (i,))
    t = _data.coords["start_time"].to_numpy()
    
    large_windows = findLargeWindow(t, window_secs=7*86400.0)
    for j, (window_beg, window_end) in enumerate(large_windows):
        print("[%d] Large window: %s - %s" % ( j, str(window_beg), str(window_end),))
    

    data[i] = _data #_data.interp(start_time=dts_daily, method="linear").rolling(start_time=7, center=True, min_periods=7).mean()



  
print("Dataset: %d" % (i,)) 
#for m in [10, 11, 12, 1, 2, 3]:
for m in [11, 12,]:# 1, 2, 3]:

    for lead_week in [0, 1, 2]:#range(_data.dims["lead_week"]):
        


        compare_data = []
        for _data in data:
            
            criteria = _data.start_time.dt.month == m
            _da = _data[varname].where(criteria).isel(lead_week=lead_week)
            _da = _da.to_numpy().flatten()
            _da = _da[np.isfinite(_da)]
            compare_data.append(_da)
       
        #print(compare_data[0]) 
        ttest = scipy.stats.ttest_ind(*compare_data)
        print("[m=%d, lead_week=%d]" % (m, lead_week,) )
        print(ttest)
       
        for d, _compare_data in enumerate(compare_data):
            print("Data %d : (mean, std) = (%f, %f). Members : %d" % (d, _compare_data.mean(), _compare_data.std(), len(_compare_data) )) 

def statPosNeg(arr, threshold=0):

    pos_threshold = np.abs(threshold)
    neg_threshold = - np.abs(threshold)

    neg = np.sum( np.isfinite(arr) & (arr < neg_threshold) )
    pos = np.sum( np.isfinite(arr) & (arr >= pos_threshold) )

    return neg, pos

print("Shortrng diff [thres=%.1f]: (neg, pos) = (%d, %d)" % (threshold, *statPosNeg(diff_data["err_shortrng"], threshold), ) )
print("Midrng diff   [thres=%.1f]: (neg, pos) = (%d, %d)" % (threshold, *statPosNeg(diff_data["err_midrng"], threshold),) )



neg_stat = diff_data.where(diff_data["err_shortrng"] < - threshold, drop=True)
pos_stat = diff_data.where(diff_data["err_shortrng"] >= threshold, drop=True)

neg_stat_time = (time_tools.toYearFraction( list(map(lambda x: pd.Timestamp(x), neg_stat.coords["start_time"].to_numpy()) )) - 0.5) % 1.0
pos_stat_time = (time_tools.toYearFraction( list(map(lambda x: pd.Timestamp(x), pos_stat.coords["start_time"].to_numpy()) )) - 0.5) % 1.0

#lead_time = ds.coords["lead_time"].to_numpy()/(3600*1e9)
    
print("Loading matplotlib...")

import matplotlib as mplt
if args.no_display:
    mplt.use("Agg")
else:
    mplt.use("TkAgg")


import matplotlib.pyplot as plt
print("done.")

figg, ax = plt.subplots(2, 1)

figg.suptitle("Pos Neg stat (short-range). Threshold = %.1f Pa" % (threshold,) )

ax[0].hist( neg_stat_time , bins=np.linspace(0.0, 1.0, 25))
ax[1].hist( pos_stat_time , bins=np.linspace(0.0, 1.0, 25))

ax[0].set_title("The time when coupling performs better")
ax[1].set_title("The time when coupling performs poorer")


figg.savefig("prediction_error_hist_in_time_%s.png" % use_months_str, dpi=300)

# ==========================
figg, ax = plt.subplots(2, 1)

figg.suptitle("Prediction stats")

bin_edges = np.linspace(-300, 300, 121)
ax[0].hist( diff_data["err_shortrng"] , bins=bin_edges)
ax[1].hist( diff_data["err_midrng"] , bins=bin_edges)


ax[0].set_title("0-5 days")
ax[1].set_title("6-10 days")

figg.savefig("prediction_error_hist_%s.png" % use_months_str, dpi=300)
if not args.no_display: 
    print("Showing figure...")
    plt.show()


# ==========================

fig, ax = plt.subplots(
    3, 1, figsize=(6, 12),
    sharex=True,
)

for i, _data in enumerate(data):

    color = ["black", "red", "blue"][i]
    
    err_growth_rate = _data["err_growth_rate"]
    err_growth_rate_mean = _data["err_growth_rate"].mean(dim="ens")
    err_growth_rate_spread = _data["err_growth_rate_spread"]
    err_midrng = _data["err_midrng"]
    err_shortrng = _data["err_shortrng"]
    start_time = _data.coords["start_time"]
 
    ax[0].plot(start_time, err_growth_rate_mean, color=color, alpha=0.5, linewidth=1)
    ax[1].plot(start_time, err_growth_rate_spread, color=color, alpha=0.5, linewidth=1)
    ax[2].plot(start_time, err_shortrng, color=color, alpha=0.5, linewidth=1)

    ax[0].scatter(start_time, err_growth_rate_mean, s=20, color=color, label="%d" % i)
    ax[1].scatter(start_time, err_growth_rate_spread, s=20, color=color, label="%d" % i)
    ax[2].scatter(start_time, err_shortrng, s=20, color=color, label="%d" % i)


for _ax in ax.flatten():
    _ax.set_xlabel("Start Time [hr]")
    _ax.grid()
    _ax.legend()

#ax.set_ylabel("RMS [Pa]")
ax[0].set_title("RMS growth rate (%d~%d) - mean" % (growth_rate_test_rng[0], growth_rate_test_rng[1]))
ax[1].set_title("RMS growth rate (%d~%d) - spread" % (growth_rate_test_rng[0], growth_rate_test_rng[1]))
ax[2].set_title("RMS 0-5 days")



if args.output != "":
    print("Saving output: ", args.output)
    fig.savefig(args.output, dpi=300)

if not args.no_display:
    print("Showing figure...")
    plt.show()

