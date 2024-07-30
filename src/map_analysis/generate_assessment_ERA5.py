from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import os.path
import os
import random

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
parser.add_argument('--months', type=int, nargs='+', help='Months to do statistics', default=[1,2,3,4,5,6,7,8,9,10,11,12])
parser.add_argument('--randomize-queue', action="store_true", help='If to randomize queue')
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


dataset_timelimits = dict(
    ECCC_GEPS5 = [ "1998-01-03", "2017-12-27"],
    ECCC_GEPS6 = [ "1998-01-07", "2017-12-26"],
)



print("Output directory: ", output_root_dir)
print("Selected lat: ", sel_lat)
print("Selected lon: ", sel_lon)
print("Varnames: ", varnames)
print("Target datasets: ", target_datasets)



ens_N = 4
time_units = "days since 1970-01-01 00:00:00"
#lead_times = [ pd.Timedelta(days=t) + pd.Timedelta(hours=12) for t in range(32) ]
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




def work(dt, pent, varname, start_times, lead_times, dataset, output_filename):
    
    result = dict(
        status="UNKNOWN",
        dt = dt,
        pent = pent,
        varname = varname,
        start_times=start_times,
        lead_times=lead_times,
        dataset=dataset,
        output_filename=output_filename,
    )
   
    id_str = "{ym:s}_pent{pent:d}".format(
        ym = dt.strftime("%Y-%m"),
        pent = pent,
    )
 
    try: 
        y = dt.year
        m = dt.month

        dataset_info = dataset_infos[dataset]
        url = dataset_info["url"]
        
        # Decide valid dates
        #test_start_date = pd.Timestamp(year=y, month=m, day=1)
        #test_end_date = test_start_date + pd.offsets.MonthBegin()
        #test_dates = pd.date_range(test_start_date, test_end_date, freq="D", inclusive="left")
        test_ds = xr.open_dataset(url % ("psl",), decode_times=True)
        dims = test_ds.dims

        if dims["M"] != ens_N:
            raise Exception("Dimension M != ens_N. M = %d and ens_N = %d" % (dims["M"], ens_N))


        test_ds = test_ds.sel(S=start_times).isel(X=0, Y=0, M=0, L=0)
        test_outcome = test_ds["psl"].to_numpy()
        valid_start_times = start_times[np.isfinite(test_outcome)]
       
         
        # Create dataset
        _tmp = dict()
        print("varname: ", varname)
        _tmp["%s_Emean" % varname] = (["time", "lat", "lon"], np.zeros((1, len(sel_lat), len(sel_lon))))
        _tmp["%s_Eabsmean" % varname] = (["time", "lat", "lon"], np.zeros((1, len(sel_lat), len(sel_lon))))
        _tmp["%s_E2mean" % varname] = (["time", "lat", "lon"], np.zeros((1, len(sel_lat), len(sel_lon))))
   
        _tmp["start_time"] = (["start_time",], valid_start_times,) 
        _tmp["lead_time"] = (["lead_time",], lead_times,)
        
        total_cnt = len(valid_start_times) * len(lead_times) * ens_N
        _tmp["total_cnt"] = (["time",], [total_cnt,],)
        

        output_ds = xr.Dataset(
            data_vars=_tmp,
            coords=dict(
                time=[dt,],
                lat=(["lat"], list(sel_lat)),
                lon=(["lon"], list(sel_lon)),
            ),
            attrs=dict(
                description="S2S forecast data RMS.",
                total_cnt = total_cnt,
            ),
        )
    
        Emean = np.zeros_like(output_ds["%s_Emean" % (varname,)])
        E2mean = np.zeros_like(output_ds["%s_E2mean" % (varname,)])
        Eabsmean = np.zeros_like(output_ds["%s_Eabsmean" % (varname,)])
                
        varname_ECCC = varname_ECCC_mapping[varname]
        varname_ERA5 = ERA5_loader.ERA5_longshortname_mapping[varname]

        for k, start_time in enumerate(valid_start_times):

            print("start_time: ", start_time)
            ds = xr.open_dataset(url % (varname_ECCC,), decode_times=True).sel(S=start_time)
            for l, lead_time in enumerate(lead_times):
     
                _ds = (ds.sel(L=lead_time))[varname_ECCC].sel(X=sel_lon, Y=sel_lat)
                
                ref_data = ERA5_loader.readERA5(start_time + lead_time - hr12, 24, varname)[varname_ERA5].isel(time=0).sel(longitude=sel_lon, latitude=sel_lat).to_numpy()

                for ens in range(ens_N):
                    
                    fcst_data = _ds.isel(M=ens).to_numpy()


                    Emean[0, :, :]     += fcst_data - ref_data
                    E2mean[0, :, :]    += (fcst_data - ref_data)**2
                    Eabsmean[0, :, :]  += np.abs(fcst_data - ref_data)


        Emean     /= total_cnt
        Eabsmean  /= total_cnt
        E2mean    /= total_cnt

        print("Total count = ", total_cnt)
        output_ds["%s_Emean" % varname][:]    = Emean
        output_ds["%s_E2mean" % varname][:]   = E2mean
        output_ds["%s_Eabsmean" % varname][:] = Eabsmean
        
        print("Output file: ", output_filename)
        output_ds.to_netcdf(output_filename, encoding=dict(
            time = dict(
                units = time_units,
            ), 
        ))

        result['status'] = "OK"

    except Exception as e:
        
        result['status'] = "ERROR"
        #traceback.print_stack()
        traceback.print_exc()
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


        if m not in args.months:
            print("Skip month %d" % (m,))
            continue
 
        test_start_date = pd.Timestamp(year=y, month=m, day=1)
        test_end_date = test_start_date + pd.offsets.MonthBegin()
        start_times = pd.date_range(test_start_date, test_end_date, freq="D", inclusive="left")

        # Need to arbirtrarily cut the start_times if it is outside
        # of the actual database
        start_times = start_times[
            ( start_times >= pd.Timestamp(dataset_timelimits[dataset][0]) )
            & ( start_times <= pd.Timestamp(dataset_timelimits[dataset][1]) )
        ]

        for pent in [0, 1, 2, 3]:#, 4, 5, 6]:
            
            lead_times = [ pd.Timedelta(days=i, hours=12) for i in range(5*pent, 5*(pent+1)) ]

            for varname in varnames:

                output_filename = os.path.join(output_dir, "fcst_error_%s_%s_pentad-%d.nc" % (varname, time_now_str, pent))
                if os.path.isfile(output_filename):
                    print("File %s exists. Skip." % (output_filename,))
                    continue
                else:
                    input_args.append((dt, pent, varname, start_times, lead_times, dataset, output_filename))
                    

failed_dates = []

with Pool(processes=nproc) as pool:

    results = pool.starmap(work, input_args)

    for i, result in enumerate(results):
        if result['status'] != 'OK':
            print('[%s, pent%d] Failed to generate output %s' % (
                result['dt'].strftime("%Y-%m-%d_%H"),
                result['pent'],
                result["output_filename"],
            ))

            failed_dates.append([result['dt'], result['pent']])


print("Tasks finished.")

print("Failed dates: ")
for i, (failed_date, pent) in enumerate(failed_dates):
    print("%d : %s, pent=%d" % (i+1, failed_date.strftime("%Y-%m-%d"), pent))



 
