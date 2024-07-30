from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import argparse


import traceback
import os
import pretty_latlon
pretty_latlon.default_fmt = "%d"

import ECCC_tools
import ERA5_loader

model_versions = ["GEPS5", "GEPS6"]

parser = argparse.ArgumentParser(
                    prog = 'make_ECCC_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

#parser.add_argument('--start-months', type=int, nargs="+", required=True)
parser.add_argument('--lead-pentads', type=int, default=6)
parser.add_argument('--days-per-pentad', type=int, default=5)
parser.add_argument('--year-rng', type=int, nargs=2, required=True)
parser.add_argument('--output-root', type=str, required=True)
parser.add_argument('--ECCC-postraw', type=str, required=True)
parser.add_argument('--ECCC-varset', type=str, required=True)
parser.add_argument('--ERA5-varset', type=str, required=True)
parser.add_argument('--ERA5-freq', type=str, required=True)
parser.add_argument('--cond-ERA5-varset', type=str, required=True)
parser.add_argument('--cond-ERA5-freq', type=str, required=True)
parser.add_argument('--cond-varname', type=str, required=True)
parser.add_argument('--cond-bin-bnds', type=float, nargs="+", default=np.arange(0, 1001, 250.0),)
parser.add_argument('--varname', type=str, required=True)
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)

output_root = args.output_root

# inclusive
year_rng = args.year_rng
days_per_pentad = args.days_per_pentad
ECCC_tools.archive_root = os.path.join("S2S", "ECCC", "data20")


ERA5_freq = args.ERA5_freq
ERA5_varset = args.ERA5_varset
ERA5_varname_long  = args.varname
ERA5_varname_short = ERA5_loader.ERA5_longshortname_mapping[ERA5_varname_long]


ECCC_postraw = args.ECCC_postraw
ECCC_varset = args.ECCC_varset
ECCC_varname_long  = args.varname
ECCC_varname_short = ECCC_tools.ECCC_longshortname_mapping[ECCC_varname_long]


cond_ERA5_freq = args.cond_ERA5_freq
cond_ERA5_varset = args.cond_ERA5_varset
cond_ERA5_varname_long  = args.cond_varname
cond_ERA5_varname_short = ERA5_loader.ERA5_longshortname_mapping[cond_ERA5_varname_long]

cond_bin_bnds = np.array(args.cond_bin_bnds)


def doJob(job_detail, detect_phase = False):


    result = dict(
        status="UNKNOWN",
        job_detail=job_detail,
    )
 
    
    try: 

        start_ym = job_detail['start_ym']
        model_version = job_detail['model_version']

        ECCC_varname  = job_detail['ECCC_varname']
        ECCC_varset   = job_detail['ECCC_varset']
        ECCC_postraw  = job_detail['ECCC_postraw']

        ERA5_varname  = job_detail['ERA5_varname']
        ERA5_varset   = job_detail['ERA5_varset']
        ERA5_freq     = job_detail['ERA5_freq']
 

        cond_ERA5_varname  = job_detail['cond_ERA5_varname']
        cond_ERA5_varset   = job_detail['cond_ERA5_varset']
        cond_ERA5_freq     = job_detail['cond_ERA5_freq']
        cond_bin_bnds  = job_detail['cond_bin_bnds']
        cond_bin_mids  = 0.5 * (cond_bin_bnds[1:] + cond_bin_bnds[:-1])

        start_year  = start_ym.year
        start_month = start_ym.month
 
      
        output_dir = os.path.join(
            output_root,
            job_detail['model_version'], 
        )

        output_file = "ECCC-S2S-condional-IVT_{model_version:s}_{varset:s}::{varname:s}_{start_ym:s}.nc".format(
            model_version = job_detail['model_version'],
            varset        = ECCC_varset,
            varname       = ECCC_varname,
            start_ym    = start_ym.strftime("%Y-%m"),
        )
        
        output_file_fullpath = os.path.join(
            output_dir,
            output_file,
        )
        
        result['output_file_fullpath'] = output_file_fullpath
 
        Path(output_dir).mkdir(parents=True, exist_ok=True)

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
        #lead_time_vec = [ pd.Timedelta(hours=h) for h in ( 1 + np.arange(number_of_lead_time) ) * 24 ]

        start_ym = pd.Timestamp(year=start_ym.year, month=start_ym.month, day=1)
        
        test_dts = pd.date_range(start_ym, start_ym + pd.DateOffset(months=1), freq="D", inclusive="left")
       
        start_times = [] 
        for dt in test_dts:
            
            model_version_date = ECCC_tools.modelVersionReforecastDateToModelVersionDate(model_version, dt)
            
            if model_version_date is None:
                continue
        
            start_times.append(dt)
    
            print("The date %s exists on ECMWF database. " % (dt.strftime("%m/%d")))
            
        if len(start_times) == 0:
            
            raise Exception("No valid start_times found.")
        
        
        aux_ds = ECCC_tools.open_dataset(ECCC_postraw, ECCC_varset, model_version, start_times[0])
        
        
        total_cnt = np.zeros((1, len(cond_bin_mids), args.lead_pentads, aux_ds.dims["latitude"], aux_ds.dims["longitude"]))
        Emean     = np.zeros((1, len(cond_bin_mids), args.lead_pentads, aux_ds.dims["latitude"], aux_ds.dims["longitude"]))
        E2mean    = np.zeros((1, len(cond_bin_mids), args.lead_pentads, aux_ds.dims["latitude"], aux_ds.dims["longitude"]))

        # This variable is used to adjust the time specifiction
        # between ECCC and ERA5.
        # For example, sst is a daily average, the first lead time is 12 hours (12Z) whereas I 
        # stored the average time as 00Z.

        # Target variable
        if ERA5_freq == "daily_mean": 
            ERA5_time_adjustment = - pd.Timedelta(hours=12)
        else:
            ERA5_time_adjustment = - pd.Timedelta(days=1)
 
        # Conditional variable
        if cond_ERA5_freq == "daily_mean": 
            cond_ERA5_time_adjustment = - pd.Timedelta(hours=12)
        else:
            cond_ERA5_time_adjustment = - pd.Timedelta(days=1)
 
        for k, start_time in enumerate(start_times):

            print("start_time: ", start_time)
        
            ds_ECCC = ECCC_tools.open_dataset(ECCC_postraw, ECCC_varset, model_version, start_time).isel(start_time=0)
            
            for p in range(args.lead_pentads):
                
                _ds_ECCC = ds_ECCC[ECCC_varname].isel(lead_time=slice(days_per_pentad*p, days_per_pentad*(p+1)))

                for lead_time in _ds_ECCC.coords["lead_time"].to_numpy():
                   
                    start_time_plus_lead_time = start_time + lead_time

                    #print("start_time_plus_lead_time = ", start_time_plus_lead_time) 
                    ref_data = ERA5_loader.open_dataset_ERA5(
                        start_time + lead_time + ERA5_time_adjustment,
                        ERA5_freq,
                        ERA5_varset,
                    )[ERA5_varname].isel(time=0)

                    # Interpolation
                    #print("ref_data: ", ref_data.coords["latitude"].to_numpy())
                    #print("_ds_ECCC: ", _ds_ECCC.coords["latitude"].to_numpy())
                    ref_data = ref_data.interp(
                        coords=dict(
                            latitude  = _ds_ECCC.coords["latitude"],
                            longitude = _ds_ECCC.coords["longitude"],
                        ),
                    )


                    # Load conditional variable data
                    cond_ref_data = ERA5_loader.open_dataset_ERA5(
                        start_time + lead_time + cond_ERA5_time_adjustment,
                        cond_ERA5_freq,
                        cond_ERA5_varset,
                    )[cond_ERA5_varname].isel(time=0)


                    cond_ref_data = cond_ref_data.interp(
                        coords=dict(
                            latitude  = _ds_ECCC.coords["latitude"],
                            longitude = _ds_ECCC.coords["longitude"],
                        ),
                    )



                
                    ref_data = ref_data.to_numpy()
                    cond_ref_data = cond_ref_data.to_numpy()

                    for number in _ds_ECCC.coords["number"]:
                        #print(_ds_ECCC) 
                        fcst_data = _ds_ECCC.sel(lead_time=lead_time, number=number).to_numpy()
                                
                        E  = fcst_data - ref_data
                        E2 = fcst_E**2

                        for j, _lat in enumerate(_ds_ECCC.coords["latitude"]):
                            for i, _lon in enumerate(_ds_ECCC.coords["longitude"]):
                                
                                for b in range(len(cond_bin_mids)):
                                    
                                    cond_l = cond_bin_bnds[b]
                                    cond_t = cond_bin_bnds[b+1]
                                    
                                    cond_tested = cond_ref_data[j, i]
                                    if cond_tested >= cond_l and cond_tested < cond_t:
                                        Emean[ 0, b, p, j, i]    += E[j, i]
                                        E2mean[0, b, p, j, i]    += E2[j, i]
                                        total_cnt[0, b, p, j, i] += 1

                                        break

                                    if b == len(cond_bin_mids)-1:
                                        print("WARNING: Some data is not within bin given. cond_tested = %f" % (cond_tested,))

        
        Emean     /= total_cnt
        E2mean    /= total_cnt
       
        _tmp = dict()
        _tmp["%s_Emean" % ECCC_varname]  = (["start_ym", "cond_bin_mid", "lead_pentad", "latitude", "longitude"], Emean)
        _tmp["%s_E2mean" % ECCC_varname]  = (["start_ym", "cond_bin_mid", "lead_pentad", "latitude", "longitude"], E2mean)
        _tmp["total_cnt"]  = (["start_ym", "cond_bin_mid", "lead_pentad", "latitude", "longitude"], cond_bin_mid)
        
        output_ds = xr.Dataset(
            data_vars=_tmp,
            coords=dict(
                start_ym=[start_ym,],
                cond_bin_mid=cond_bin_mids,
                cond_bin_bnds=cond_bin_bnds,
                latitude=aux_ds.coords["latitude"],
                longitude=aux_ds.coords["longitude"],
            ),
            attrs=dict(
                description="S2S forecast data RMS.",
                cond_ERA5_varname=cond_ERA5_varname,
                cond_ERA5_varset=cond_ERA5_varset,
                cond_ERA5_freq=cond_ERA5_freq,
            ),
        )
    

        print("Output file: ", output_file_fullpath)
        output_ds.to_netcdf(output_file_fullpath)

        result['status'] = "OK"

    except Exception as e:
        
        result['status'] = "ERROR"
        #traceback.print_stack()
        traceback.print_exc()
        print(e)


    return result



failed_dates = []
input_args = []


start_yms = pd.date_range(
    pd.Timestamp(year=year_rng[0], month=1,  day=1),
    pd.Timestamp(year=year_rng[1]+1, month=1, day=1),
    freq="M",
    inclusive="both",
)

for model_version in model_versions:
    
    print("[MODEL VERSION]: ", model_version)
    
    for start_ym in start_yms:

        if start_ym.month in [4, 5, 6, 7, 8, 9, 10, ]:
            print("For now skip doing this month: ", str(start_ym))
            continue
 
        job_detail = dict(
            start_ym      = start_ym,
            model_version = model_version,
            ECCC_postraw  = ECCC_postraw,
            ECCC_varset   = ECCC_varset,
            ECCC_varname  = ECCC_varname_short,
            ERA5_varset   = ERA5_varset,
            ERA5_varname  = ERA5_varname_short,
            ERA5_freq     = ERA5_freq,
            cond_ERA5_varset   = ERA5_varset,
            cond_ERA5_varname  = ERA5_varname_short,
            cond_ERA5_freq     = ERA5_freq,
            cond_bin_bnds      = cond_bin_bnds,
        )
        
        print("[Detect] Checking year-month = %s" % (start_ym.strftime("%Y-%m"),))
        
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

