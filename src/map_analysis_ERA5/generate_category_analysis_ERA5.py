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
parser.add_argument('--varname', type=str, required=True)
parser.add_argument('--nproc', type=int, default=1)
parser.add_argument('--catvar-postraw', type=str, required=True)
parser.add_argument('--catvar-varname', type=str, required=True)
parser.add_argument('--catvar-varset', type=str, required=True)
parser.add_argument('--catvar-bnds', type=float, nargs="+", required=True)
args = parser.parse_args()
print(args)

output_root = args.output_root

# Test categories


catvar_bnds = np.array(args.catvar_bnds) 
dval = catvar_bnds[1:] - catvar_bnds[:-1]
if np.any( dval < 0 ):
    raise Exception("Error: `--var-bnds` should be monotonically increasing.")

number_of_categories = len(dval)

catvar_postraw = args.catvar_postraw
catvar_varset = args.catvar_varset
catvar_varname_long  = args.catvar_varname
catvar_varname_short = ECCC_tools.ECCC_longshortname_mapping[catvar_varname_long]


print("The value of catvar_bnds: ", catvar_bnds)

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

        start_year  = start_ym.year
        start_month = start_ym.month
 
      
        output_dir = os.path.join(
            output_root,
            job_detail['model_version'], 
        )

        output_file = "ECCC-S2S_{model_version:s}_{varset:s}::{varname:s}_{start_ym:s}.nc".format(
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
        
        total_cnt = np.zeros((1, args.lead_pentads, number_of_categories, aux_ds.dims["latitude"], aux_ds.dims["longitude"]))
        Emean     = np.zeros((1, args.lead_pentads, number_of_categories, aux_ds.dims["latitude"], aux_ds.dims["longitude"]))
        E2mean    = np.zeros((1, args.lead_pentads, number_of_categories, aux_ds.dims["latitude"], aux_ds.dims["longitude"]))

        Cmean     = np.zeros((1, args.lead_pentads, number_of_categories, aux_ds.dims["latitude"], aux_ds.dims["longitude"]))
        C2mean    = np.zeros((1, args.lead_pentads, number_of_categories, aux_ds.dims["latitude"], aux_ds.dims["longitude"]))


        # This variable is used to adjust the time specifiction
        # between ECCC and ERA5.
        # For example, sst is a daily average, the first lead time is 12 hours (12Z) whereas I 
        # stored the average time as 00Z.

        if ERA5_freq == "daily_mean": 
            ERA5_time_adjustment = - pd.Timedelta(hours=12)
        else:
            ERA5_time_adjustment = - pd.Timedelta(days=1)
 
        for k, start_time in enumerate(start_times):

            print("start_time: ", start_time)
        
            ds_ECCC = ECCC_tools.open_dataset(ECCC_postraw, ECCC_varset, model_version, start_time).isel(start_time=0)
            ds_cat  = ECCC_tools.open_dataset(catvar_postraw, catvar_varset, model_version, start_time).isel(start_time=0)
            
            for p in range(args.lead_pentads):
                
                _ds_ECCC = ds_ECCC[ECCC_varname].isel(lead_time=slice(days_per_pentad*p, days_per_pentad*(p+1)))
                _ds_cat  = ds_cat[catvar_varname_long].isel(lead_time=slice(days_per_pentad*p, days_per_pentad*(p+1)))

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

                    ref_data = ref_data.to_numpy()

                    for number in _ds_ECCC.coords["number"]:
                        
                        fcst_data = _ds_ECCC.sel(lead_time=lead_time, number=number).to_numpy()
                        catvar_data = _ds_cat.sel(lead_time=lead_time, number=number).to_numpy()

                        _diff = fcst_data - ref_data
                        
                        for j in range(len(_ds_ECCC.coords["latitude"])):
                            for i in range(len(_ds_ECCC.coords["longitude"])):
                                
                                for b in range(number_of_categories):
                                
                                    catvar_bnd_left  = catvar_bnds[b]
                                    catvar_bnd_right = catvar_bnds[b+1]
                                    
                                    if ( catvar_data[j, i] >= catvar_bnd_left ) and (catvar_data[j, i] < catvar_bnd_right) :
                                        
                                        Emean[0, p, b, j, i]     += _diff[j, i]
                                        E2mean[0, p, b, j, i]    += _diff[j, i]**2

                                        Cmean[0, p, b, j, i]     += catvar_data[j, i]
                                        C2mean[0, p, b, j, i]    += catvar_data[j, i]**2

                                        total_cnt[0, p, b, j, i] += 1
                                        
                                        break

                                    elif b == number_of_categories - 1:
                                        print("Warning: cannot find category. Value: %f" % (catvar_data[j, i],) )
                                        print("Warning: catvar_bnds = ", catvar_bnds)


        expected_total_cnt = (
            len(start_times) 
            * ( days_per_pentad * args.lead_pentads ) 
            * len(aux_ds.coords["number"]) 
            * len(aux_ds.coords["latitude"])
            * len(aux_ds.coords["longitude"]) 
        )
        sum_total_cnt = np.sum(total_cnt)

        if sum_total_cnt != expected_total_cnt:
            print("Warning: expected_total_cnt does not match sum_total_cnt. ")
            print("Warning: expected_total_cnt = %d , sum_total_cnt = %d" % (expected_total_cnt, sum_total_cnt))
            print("Warning: (start_times, lead_time, number, lat, lon) = (%d, %d, %d, %d, %d)" % (
                len(start_times),
                len(aux_ds.coords["lead_time"]),
                len(aux_ds.coords["number"]),
                len(aux_ds.coords["latitude"]),
                len(aux_ds.coords["longitude"]),
            )) 

        Emean     /= total_cnt
        E2mean    /= total_cnt
        Cmean     /= total_cnt
        C2mean    /= total_cnt
        
        print("Total count = ", total_cnt)
        
        _tmp = dict()
        _tmp["%s_Emean" %  ECCC_varname]  = (["start_ym", "lead_pentad", "category", "latitude", "longitude"], Emean)
        _tmp["%s_E2mean" % ECCC_varname]  = (["start_ym", "lead_pentad", "category", "latitude", "longitude"], E2mean)
        _tmp["%s_Cmean" %  ECCC_varname]  = (["start_ym", "lead_pentad", "category", "latitude", "longitude"], Cmean)
        _tmp["%s_C2mean" % ECCC_varname]  = (["start_ym", "lead_pentad", "category", "latitude", "longitude"], C2mean)
        _tmp["total_cnt"] = (["start_ym", "lead_pentad", "category", "latitude", "longitude"], total_cnt,)
        _tmp["catvar_bnds"] = (["category_bnd"], catvar_bnds,)
        
        output_ds = xr.Dataset(
            data_vars=_tmp,
            coords=dict(
                start_ym=[start_ym,],
                latitude=aux_ds.coords["latitude"],
                longitude=aux_ds.coords["longitude"],
                category=list(range(number_of_categories)),
                category_bnds=list(range(len(catvar_bnds))),
            ),
            attrs=dict(
                description="S2S forecast data RMS.",
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

