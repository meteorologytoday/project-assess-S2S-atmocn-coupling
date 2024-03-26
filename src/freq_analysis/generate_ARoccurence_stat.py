from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import argparse

import ECCC_tools
import traceback
import os
import pretty_latlon
pretty_latlon.default_fmt = "%d"

import time


def computePearsonCorrelation(xa, ya):
    
    xa_m = np.mean(xa)
    ya_m = np.mean(ya)

    xa_a = xa - xa_m
    ya_a = ya - ya_m

    a = np.sum( xa_a * ya_a )
    b = np.sqrt( np.sum(xa_a**2) * np.sum(ya_a**2) )

    return a / b


model_versions = ["GEPS5", "GEPS6"]

parser = argparse.ArgumentParser(
                    prog = 'make_ECCC_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

parser.add_argument('--input-dir', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--region', type=str, required=True)
parser.add_argument('--year-rng', type=int, nargs=2, required=True)
parser.add_argument('--start-time-months', type=int, nargs='+', required=True)
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)


beg_year = args.year_rng[0]
end_year = args.year_rng[1]
number_of_years = end_year - beg_year + 1

number_of_lead_time = 32
days_per_week = 7
number_of_weeks = 4


# inclusive
year_rng = args.year_rng

def open_ARoccur_timeseries_dataset(model_version, region, dt):
   
    varset = "ARoccur_timeseries"
    filename = os.path.join(
        args.input_dir,
        model_version,
        varset,
        region,
        "ECCC-S2S_{model_version:s}_{varset:s}_{time_str:s}.nc".format(
            model_version = model_version,
            varset = varset,
            time_str = dt.strftime("%Y_%m-%d"),
        )
    ) 

    return xr.open_dataset(filename)


def doJob(job_detail, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(job_detail=job_detail, status="UNKNOWN", need_work=False, detect_phase=detect_phase, output_file_fullpath=None)

    output_varset = "ARoccur-stat"
    try:
        
        start_months = job_detail['start_months']
        model_version = job_detail['model_version']
       
        start_months_str = "-".join(["%02d" % m for m in start_months])
        print("Start months: ", start_months)

        start_mds = []

        # Check available start_time
        dts_in_year = pd.date_range("2021-01-01", "2021-12-31", inclusive="both")
        for dt in dts_in_year:
           
            if not (dt.month in start_months):
                continue
 
            model_version_date = ECCC_tools.modelVersionReforecastDateToModelVersionDate(model_version, dt)

            if model_version_date is None:
                continue

            print("The date %s exists on ECMWF database. " % (dt.strftime("%m/%d")))
            start_mds.append((dt.month, dt.day))

        number_of_samples_per_week = len(start_mds) * number_of_years
        data = np.zeros((number_of_weeks, number_of_samples_per_week, 2))
        stat_data = np.zeros((number_of_weeks,))

        output_file = "ECCC-S2S_{model_version:s}_ARoccur-stat_{start_months:s}.nc".format(
            model_version = job_detail['model_version'],
            start_months  = start_months_str,
        )

        output_file_fullpath = os.path.join(
            args.output_dir,
            output_file,
        )
        
        result['output_file_fullpath'] = output_file_fullpath
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # First round is just to decide which files
        # to be processed to enhance parallel job 
        # distribution. I use variable `phase` to label
        # this stage.
        file_exists = os.path.isfile(output_file_fullpath)

        if detect_phase is True:
            result['need_work'] = not file_exists
            result['status'] = 'OK' 
            return result


        # Load file

        cnt = 0
        for start_year in range(beg_year, end_year+1):
            for start_month, start_day in start_mds:

                dt = pd.Timestamp(year=start_year, month=start_month, day=start_day)
                ds = open_ARoccur_timeseries_dataset(model_version, args.region, dt)

                da_ARoccur_ECCC = ds["ARoccur_ECCC"].mean(dim="number")
                ARoccur_anom_ECCC = da_ARoccur_ECCC - ds["ARoccur_clim"]
                ARoccur_anom_reanalysis = ds["ARoccur_reanalysis"] - ds["ARoccur_clim"]

                for week in range(number_of_weeks):
                    
                    sel_idx = slice(week * days_per_week, (week+1) * days_per_week)
                    data[week, cnt, 0] = np.mean(ARoccur_anom_ECCC[0, sel_idx])
                    data[week, cnt, 1] = np.mean(ARoccur_anom_reanalysis[0, sel_idx])

                cnt += 1

        for week in range(number_of_weeks):
            print("ts0: ", data[week, :, 0])
            print("ts1: ", data[week, :, 1])
            stat_data[week] = computePearsonCorrelation(data[week, :, 0], data[week, :, 1])

        ds_new = xr.Dataset(

            data_vars=dict(
                ARoccur_corr=(["month_group", "week",], np.reshape(stat_data, (1, number_of_weeks))),
            ),

            coords=dict(
                week=(["week",], np.arange(number_of_weeks)),
                month_group=[start_months_str],
            ),
        )
        
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
     
        print("Writing to file: %s" % (output_file_fullpath,) )
        ds_new.to_netcdf(
            output_file_fullpath,
            unlimited_dims=["month_group",],
        )

        result['status'] = 'OK'

    except Exception as e:

        print("Error. Now print stacktrace...")
        import traceback
        traceback.print_exc()


    return result



failed_dates = []

#dts_in_year = pd.date_range("2021-01-31", "2021-01-31", inclusive="both")
input_args = []
for model_version in model_versions:
    
    print("[MODEL VERSION]: ", model_version)
    
    print("[Detect] Checking start_months:", args.start_time_months)
    job_detail = dict(
        model_version = model_version,
        start_months = args.start_time_months,
    )


    result = doJob(job_detail, detect_phase=True)

    if not result['need_work']:
        print("File `%s` already exist. Skip it." % (result['output_file_fullpath'],))
        continue
    

    input_args.append((job_detail, False))

                
with Pool(processes=args.nproc) as pool:

    results = pool.starmap(doJob, input_args)

    for i, result in enumerate(results):
        if result['status'] != 'OK':
            print('!!! Failed to generate output file %s.' % (result['output_file_fullpath'],))
            failed_dates.append(result['job_detail'])


print("Tasks finished.")

print("Failed output files: ")
for i, failed_detail in enumerate(failed_dates):
    print("%d : " % (i+1), failed_detail)

print("Done.")

