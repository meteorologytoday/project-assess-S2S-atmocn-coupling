from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import os.path
import os

import xarray as xr
import numpy as np
import pandas as pd
import ERA5_loader
import traceback

import pretty_latlon
pretty_latlon.default_fmt = "%d"

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--lat-rng', type=float, nargs=2, help='Lat range.', required=True)
parser.add_argument('--lon-rng', type=float, nargs=2, help='Lon range.', required=True)
parser.add_argument('--beg-date', type=str, help='Range of time. Format: yyyy-mm', required=True)
parser.add_argument('--end-date', type=str, help='Range of time. Format: yyyy-mm', required=True)
parser.add_argument('--nproc', type=int, default=5)
args = parser.parse_args()




nproc = args.nproc

beg_lat = args.lat_rng[0]
end_lat = args.lat_rng[1]

beg_lon = args.lon_rng[0] % 360.0
end_lon = args.lon_rng[1] % 360.0

sel_lat = np.arange(beg_lat, end_lat + 0.5, 1)
sel_lon = np.arange(beg_lon, end_lon + 0.5, 1)


output_root_dir = "./output_fcst_error_%s-%s_%s-%s" % (
    pretty_latlon.pretty_lat(beg_lat), pretty_latlon.pretty_lat(end_lat),
    pretty_latlon.pretty_lon(beg_lon), pretty_latlon.pretty_lon(end_lon),
)

#sel_lat = np.arange(30, 40.5, 1)
#sel_lon = np.arange(225, 235.5, 1)
varnames = [
    "mean_sea_level_pressure",
#    "10m_u_component_of_wind",
#    "10m_v_component_of_wind",
]

target_year_months = pd.date_range(
    start=args.beg_date,
    end=args.end_date,
    freq="MS",
)

target_datasets = [
    "ECCC_GEPS5",
    "ECCC_GEPS6",
]


print("Output directory: ", output_root_dir)
print("Selected lat: ", sel_lat)
print("Selected lon: ", sel_lon)
print("Varnames: ", varnames)
print("Target datasets: ", target_datasets)



ens_N = 4
time_units = "days since 1970-01-01 00:00:00"
lead_times = [ pd.Timedelta(days=t) + pd.Timedelta(hours=12) for t in range(32) ]
#lead_times = [ pd.Timedelta(days=t) + pd.Timedelta(hours=12) for t in range(1) ]

dataset_infos = dict(
    
    ECCC_GEPS5 = dict(
        url = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS5/.hindcast/%s/dods",
    ),

    ECCC_GEPS6 = dict(
        url = "http://iridl.ldeo.columbia.edu/SOURCES/.Models/.SubX/.ECCC/.GEPS6/.hindcast/%s/dods",
    ),
)

varname_ECCC_mapping = {
    "mean_sea_level_pressure" : "psl",
    "10m_u_component_of_wind" : "uas",
    "10m_v_component_of_wind" : "vas",
}

hr12 = pd.Timedelta(hours=12)




def work(dt, dataset, output_filename):
    
    result = dict(status="UNKNOWN", dt=dt, dataset=dataset, output_filename=output_filename,)

   
    try: 
        y = dt.year
        m = dt.month

        dataset_info = dataset_infos[dataset]
        url = dataset_info["url"]    
        
        # Decide valid dates
        test_start_date = pd.Timestamp(year=y, month=m, day=1)
        test_end_date = test_start_date + pd.offsets.MonthBegin()
        test_dates = pd.date_range(test_start_date, test_end_date, freq="D", inclusive="left")
        test_ds = xr.open_dataset(url % ("psl",), decode_times=True).sel(S=test_dates).isel(X=0, Y=0, M=0, L=0)
        test_outcome = test_ds["psl"].to_numpy()
        model_start_times = test_dates[np.isfinite(test_outcome)]

        # Create dataset
        _tmp = dict()
        for varname in varnames:
            print("varname: ", varname)
            _tmp["%s_RMS" % varname] = (["ens", "start_time", "lead_time"], np.zeros((ens_N, len(model_start_times), len(lead_times))) )
            _tmp["%s_MAXDIFF" % varname] = (["ens", "start_time", "lead_time"], np.zeros((ens_N, len(model_start_times), len(lead_times))) )
        
        output_ds = xr.Dataset(
            data_vars=_tmp,
            coords=dict(
                ens=np.arange(ens_N) + 1,
                start_time=model_start_times,
                lead_time=lead_times,
            ),
            attrs=dict(description="S2S forecast data RMS."),
        )
    
        for varname in varnames:
                
            print("varname: ", varname)
     
            rms = np.zeros((ens_N, len(model_start_times), len(lead_times)))
            maxdiff = rms.copy()
            

                    
            varname_ECCC = varname_ECCC_mapping[varname]
            varname_ERA5 = ERA5_loader.ERA5_longshortname_mapping[varname]

            for k, start_time in enumerate(model_start_times):
                print("start_time: ", start_time)
                ds = xr.open_dataset(url % (varname_ECCC,), decode_times=True).sel(S=start_time)
                
                for l, lead_time in enumerate(lead_times):
         
                    _ds = (ds.sel(L=lead_time))[varname_ECCC].sel(X=sel_lon, Y=sel_lat)
                    
                    ref_data = ERA5_loader.readERA5(start_time + lead_time - hr12, 24, varname)[varname_ERA5].isel(time=0).sel(longitude=sel_lon, latitude=sel_lat).to_numpy()

                    for ens in range(ens_N):
                        
                        fcst_data = _ds.isel(M=ens).to_numpy()
                        rms[ens, k, l]     = np.std(fcst_data - ref_data)
                        maxdiff[ens, k, l] = np.amax(np.abs(fcst_data - ref_data))


            output_ds["%s_RMS" % varname][:] = rms
            output_ds["%s_MAXDIFF" % varname][:] = maxdiff

            print("Output file: ", output_filename)
            output_ds.to_netcdf(output_filename, encoding=dict(
                start_time = dict(
                    units = time_units,
                ), 
            ))

            result['status'] = "OK"

    except Exception as e:
        
        result['status'] = "ERROR"
        traceback.print_stack()
        print(e)


    return result


input_args = [] 
for dataset in target_datasets:
        
    output_dir = os.path.join(output_root_dir, dataset)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for dt in target_year_months:

        y = dt.year
        m = dt.month
        
        time_now_str = dt.strftime("%Y-%m")
        output_filename = os.path.join(output_dir, "fcst_error_%s.nc" % (time_now_str,))
        
        if os.path.isfile(output_filename):
            print("File %s exists. Skip." % (output_filename,))
            continue
        else:
            input_args.append((dt, dataset, output_filename))
            


failed_dates = []

with Pool(processes=nproc) as pool:

    results = pool.starmap(work, input_args)

    for i, result in enumerate(results):
        if result['status'] != 'OK':
            print('[%s] Failed to generate output %s' % (
                result['dt'].strftime("%Y-%m-%d_%H"),
                result["output_filename"],
            ))

            failed_dates.append(result['dt'])


print("Tasks finished.")

print("Failed dates: ")
for i, failed_date in enumerate(failed_dates):
    print("%d : %s" % (i+1, failed_date.strftime("%Y-%m-%d"),))



 
