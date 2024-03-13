import xarray as xr
import numpy as np
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
#varname = "10m_u_component_of_wind"

beg_ym_parsed = args.year_rng[0].split("-")
end_ym_parsed = args.year_rng[1].split("-")


beg_ym = pd.Timestamp(year=int(beg_ym_parsed[0]), month=int(beg_ym_parsed[1]), day=1)
end_ym = pd.Timestamp(year=int(end_ym_parsed[0]), month=int(end_ym_parsed[1]), day=1)

dts = pd.date_range(start=beg_ym, end=end_ym, freq="MS")


dts_daily = pd.date_range(start=beg_ym, end=end_ym, freq="D")


growth_rate_test_rng = [0, 5]


data = []
ref_time = pd.Timestamp("1970-01-01 00:00:00")

for input_dir in args.input_dirs:
    filenames = []
    for dt in dts:
        filenames.append(
            os.path.join(input_dir, "fcst_error_%s.nc" % (dt.strftime("%Y-%m"),))
        )
  
    print(filenames) 
    ds = xr.open_mfdataset(filenames, concat_dim="start_time", combine="nested")


    # Now processing data
    _err_growth_rate = np.zeros((ds.dims["ens"], ds.dims["start_time"]))
    _err_growth_rate_spread = np.zeros((ds.dims["start_time"],))
    
    _err_midrng = np.zeros((ds.dims["start_time"],))
    _err_shortrng = np.zeros((ds.dims["start_time"],))
        
    for j, start_time in enumerate(ds.coords["start_time"]):
        for i, ens in enumerate(ds.coords["ens"]):
            _fcst_err = ds[varname].sel(ens=ens, start_time=start_time).isel(lead_time=slice(growth_rate_test_rng[0], growth_rate_test_rng[1])).to_numpy()
            _fit = np.polyfit(np.arange(len(_fcst_err)), _fcst_err, 1)
            _err_growth_rate[i, j] = _fit[0]
            
            
        _err_growth_rate_spread[j] = np.std(_err_growth_rate[:, j], ddof=1)
            

        _err_shortrng[j] = ds[varname].sel(start_time=start_time).isel(lead_time=slice(0, 5)).mean(dim=["ens", "lead_time"]).to_numpy()
        _err_midrng[j] = ds[varname].sel(start_time=start_time).isel(lead_time=slice(5, 10)).mean(dim=["ens", "lead_time"]).to_numpy()

    
    _err_growth_rate_mean = np.mean(_err_growth_rate, axis=0)

    _ds = xr.Dataset(
        data_vars=dict(
            err_growth_rate = (["ens", "start_time"], _err_growth_rate),
            err_growth_rate_spread = (["start_time"], _err_growth_rate_spread),
            err_shortrng = (["start_time"], _err_shortrng),
            err_midrng   = (["start_time"], _err_midrng),
        ),
        coords=dict(
            start_time=ds.coords["start_time"].to_numpy(),
            reference_time=ref_time,
        ),
    )


    data.append(_ds)


for i, _data in enumerate(data):

    print("Interpolating dataset %d" % (i,))
    t = _data.coords["start_time"].to_numpy()
    
    large_windows = findLargeWindow(t, window_secs=7*86400.0)
    for j, (window_beg, window_end) in enumerate(large_windows):
        print("[%d] Large window: %s - %s" % ( j, str(window_beg), str(window_end),))
    

    data[i] = _data.interp(start_time=dts_daily, method="linear").rolling(start_time=7, center=True, min_periods=7).mean()


use_months = np.unique([10, 11, 12, 1, 2, 3, 4, 5, 6])
use_months = np.unique([1, 2, 3])
use_months_str = ["%02d" % _m  for _m in use_months ]
use_months_str = "-".join(use_months_str)
print(use_months_str)
diff_data = data[1] - data[0]

criteria = None
for use_month in use_months:
    if criteria is None:
        criteria = diff_data.start_time.dt.month == use_month
    else:
        criteria = criteria | (diff_data.start_time.dt.month == use_month)

        
if criteria is not None:
    diff_data = diff_data.where(criteria, drop=True)
 
print("Average shortrng diff : ", diff_data["err_shortrng"].mean(skipna=True).to_numpy(), ", std: ", diff_data["err_shortrng"].std(skipna=True).to_numpy())
print("Average midrng diff : ", diff_data["err_midrng"].mean(skipna=True).to_numpy(), ", std: ", diff_data["err_midrng"].std(skipna=True).to_numpy())

threshold = 10

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

