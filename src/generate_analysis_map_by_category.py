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
ECCC_tools.init()

import ERA5_loader



model_versions = ["GEPS6sub1", "GEPS5",]# "GEPS6sub2", "GEPS6"]

parser = argparse.ArgumentParser(
                    prog = 'make_ECCC_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

#parser.add_argument('--start-months', type=int, nargs="+", required=True)
parser.add_argument('--category-file', type=str, required=True)
parser.add_argument('--date-to-category-file', type=str, required=True)
parser.add_argument('--category-name', type=str, default="category")
parser.add_argument('--ignored-categories', type=str, nargs="*", default=[])

parser.add_argument('--lead-windows', type=int, default=6)
parser.add_argument('--days-per-window', type=int, default=5)

#parser.add_argument('--year-rng', type=int, nargs=2, required=True)
parser.add_argument('--output-root', type=str, required=True)
parser.add_argument('--ECCC-postraw', type=str, required=True)
parser.add_argument('--ECCC-varset', type=str, required=True)
parser.add_argument('--ERA5-varset', type=str, required=True)
parser.add_argument('--ERA5-freq', type=str, required=True)
parser.add_argument('--levels', nargs="+", type=int, help="If variable is 3D.", default=None)
parser.add_argument('--varname', type=str, required=True)
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)

output_root = args.output_root

# inclusive
days_per_window = args.days_per_window
ECCC_tools.archive_root = os.path.join("S2S", "ECCC", "data20_20240723")


ERA5_freq = args.ERA5_freq
ERA5_varset = args.ERA5_varset
ERA5_varname_long  = args.varname
ERA5_varname_short = ERA5_loader.ERA5_longshortname_mapping[ERA5_varname_long]


ECCC_postraw = args.ECCC_postraw
ECCC_varset = args.ECCC_varset
ECCC_varname_long  = args.varname
ECCC_varname_short = ECCC_tools.ECCC_longshortname_mapping[ECCC_varname_long]



print("Category column named as `%s`" % (args.category_name,))
print("Category file: ", args.category_file)
print("Ignored catories: ", args.ignored_categories)

categories = pd.read_csv(args.category_file)[args.category_name].to_numpy()
_tmp = []
for category in categories:
    if category not in args.ignored_categories:
        _tmp.append(category)

categories = _tmp
date_category_mapping = pd.read_csv(args.date_to_category_file)[['date', args.category_name]]

print("I read %d categories: %s." % (
    len(categories),
    ", ".join([str(category) for category in categories])
))




def doJob(job_detail, detect_phase = False):


    result = dict(
        status="UNKNOWN",
        job_detail=job_detail,
    )
 
    
    try: 

        category = job_detail['category']
        category_name = job_detail['category_name']

        model_version = job_detail['model_version']

        ECCC_varname  = job_detail['ECCC_varname']
        ECCC_varset   = job_detail['ECCC_varset']
        ECCC_postraw  = job_detail['ECCC_postraw']

        ERA5_varname  = job_detail['ERA5_varname']
        ERA5_varset   = job_detail['ERA5_varset']
        ERA5_freq     = job_detail['ERA5_freq']

        output_dir = os.path.join(
            output_root,
            job_detail['model_version'], 
        )

        output_file = "ECCC-S2S_{model_version:s}_{varset:s}::{varname:s}_{category_name:s}-{category:s}.nc".format(
            model_version = job_detail['model_version'],
            varset        = ECCC_varset,
            varname       = ECCC_varname,
            category      = str(category),
            category_name = category_name,
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


        print("Finding dates cateogry: %s" % (str(category),)) 
        # these selected dts are start_time
        test_dts = pd.to_datetime(date_category_mapping.loc[date_category_mapping[category_name] == category, "date"])
       
        start_times = [] 
        for dt in test_dts:
            
            model_version_date = ECCC_tools.modelVersionReforecastDateToModelVersionDate(model_version, dt)
            
            if model_version_date is None:
                continue


            print("Found valid time: ", dt.strftime("%Y-%m-%d"))
            start_times.append(dt)
    
            print("The date %s exists on ECMWF database. " % (dt.strftime("%m/%d")))
            
        if len(start_times) == 0:
            
            raise Exception("No valid start_times found for dt = %s." % (str(dt),))
        
        
        aux_ds = ECCC_tools.open_dataset(ECCC_postraw, ECCC_varset, model_version, start_times[0])

        print(aux_ds)

        variable3D = "level" in aux_ds[ECCC_varname].dims
        do_level_sel = variable3D and args.levels is not None
            

        
        dim_cnt = (1, args.lead_windows)
        if variable3D:
            if args.levels is None:
                nlev = aux_ds.dims["level"]

            else:
                nlev = len(args.levels)

            dim_E   = (1, args.lead_windows, nlev, aux_ds.dims["latitude"], aux_ds.dims["longitude"])
        else:
            dim_E   = (1, args.lead_windows, aux_ds.dims["latitude"], aux_ds.dims["longitude"])
        
        # total_cnt means the number of lead days and ensemble members at each start time.   
        total_cnt = np.zeros(dim_cnt)
        ddof      = np.zeros(dim_cnt)
        Emean     = np.zeros(dim_E)
        E2mean    = np.zeros(dim_E)

        Eabsmean  = np.zeros(dim_E)
        Eabs2mean = np.zeros(dim_E)


        # This variable is used to adjust the time specifiction
        # between ECCC and ERA5.
        # For example, sst is a daily average, the first lead time is 12 hours (12Z) whereas I 
        # stored the average time as 00Z.

        if ERA5_freq == "daily_mean": 
            # 2025/01/10
            # By matching the sensible and latent heat flux map,
            # I think download daily ERA5 data by month puts the
            # average of day XXXX-01-01T00:00 ~ XXXX-01-02T00:00 in
            # the time XXXX-01-01T00:00
            ERA5_time_adjustment = - pd.Timedelta(hours=12) 
        elif ERA5_freq == "daily_acc": 
            # 2025/01/10
            # By matching the sensible and latent heat flux map,
            # I think download daily ERA5 data by month puts the
            # average of day XXXX-01-01T00:00 ~ XXXX-01-02T00:00 in
            # the time XXXX-01-01T00:00
            ERA5_time_adjustment = - pd.Timedelta(hours=12) 

        elif ERA5_freq == "inst":
            ERA5_time_adjustment = - pd.Timedelta(hours=24)
        else:
            raise Exception("Unknown ERA5_freq %s" % (ERA5_freq,))
 
        for k, start_time in enumerate(start_times):

            print("start_time: ", start_time)
        
            ds_ECCC = ECCC_tools.open_dataset(ECCC_postraw, ECCC_varset, model_version, start_time).isel(start_time=0)
            
            for p in range(args.lead_windows):
                
                _ds_ECCC = ds_ECCC[ECCC_varname].isel(lead_time=slice(days_per_window*p, days_per_window*(p+1)))

                if do_level_sel:
                    _ds_ECCC = _ds_ECCC.sel(level=args.levels)

                for l, lead_time in enumerate(_ds_ECCC.coords["lead_time"].to_numpy()):
                   
                    #start_time_plus_lead_time = start_time + lead_time

                    #print("start_time_plus_lead_time = ", start_time_plus_lead_time) 
                    ref_data = ERA5_loader.open_dataset_ERA5(
                        start_time + lead_time + ERA5_time_adjustment,
                        ERA5_freq,
                        ERA5_varset,
                    )[ERA5_varname].isel(time=0)


                    if ERA5_varname_long in [ "mean_surface_sensible_heat_flux", "mean_surface_latent_heat_flux" ]:
                        reverse_sign = -1
                        print("Var %s need to reverse sign. Now multiply it by %d. " % (ERA5_varname_long, reverse_sign,))
                        ref_data *= reverse_sign

                    if do_level_sel:
                        ref_data = ref_data.sel(level=args.levels)


                    if args.varname == "geopotential":
                        ref_data /= 9.81 
                    elif ERA5_varname_long == "total_precipitation":
                        print("Var %s need to convert from meter to milli-meter. " % (ERA5_varname_long,))
                        ref_data *= 1e3
 
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

                        fcst_error = fcst_data - ref_data
                        Emean[0, p]     += fcst_error
                        E2mean[0, p]    += fcst_error**2
                        Eabsmean[0, p]  += np.abs(fcst_error)
                        Eabs2mean[0, p] += np.abs(fcst_error)**2
                        total_cnt[0, p] += 1

                        # I consider two days of the same prediction dependent. 
                        if l == 0:
                            ddof[0, p] += 1


        if variable3D:
            ttl_cnt_ext = total_cnt[:, :, None, None, None]
        else:
            ttl_cnt_ext = total_cnt[:, :, None, None]
            
        Emean     /= ttl_cnt_ext
        E2mean    /= ttl_cnt_ext
        Eabsmean  /= ttl_cnt_ext
        Eabs2mean  /= ttl_cnt_ext
        print("Total count = ", total_cnt)
        

        # prepping for output
 
        coords=dict(
            category=[category,],
            latitude=aux_ds.coords["latitude"],
            longitude=aux_ds.coords["longitude"],
        )

        if variable3D:
            dim_E   = [args.category_name, "lead_window", "level", "latitude", "longitude"] 
            if args.levels is None:
                coords["level"] = aux_ds.coords["level"]
            else:
                coords["level"] = (["level",], np.array(args.levels, dtype=float))
        

        else:
            dim_E   = [args.category_name, "lead_window", "latitude", "longitude"]

        _tmp = dict()
        _tmp["%s_Emean" % ECCC_varname]  = (dim_E, Emean)
        _tmp["%s_E2mean" % ECCC_varname]  = (dim_E, E2mean)
        _tmp["%s_Eabsmean" % ECCC_varname]  = (dim_E, Eabsmean)
        _tmp["%s_Eabs2mean" % ECCC_varname]  = (dim_E, Eabs2mean)

        _tmp["total_cnt"] = ([args.category_name, "lead_window"], total_cnt,)
        _tmp["ddof"] = ([args.category_name, "lead_window"], ddof,)
 
        output_ds = xr.Dataset(
            data_vars=_tmp,
            coords=coords,
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


for model_version in model_versions:
    
    print("[MODEL VERSION]: ", model_version)
    
    for category in categories:

        #if start_ym.month not in [4,]:
        #    print("For now skip doing this month: ", str(start_ym))
        #    continue
 
        job_detail = dict(
            category      = category,
            category_name = args.category_name,
            model_version = model_version,
            ECCC_postraw  = ECCC_postraw,
            ECCC_varset   = ECCC_varset,
            ECCC_varname  = ECCC_varname_short,
            ERA5_varset   = ERA5_varset,
            ERA5_varname  = ERA5_varname_short,
            ERA5_freq     = ERA5_freq,
        )
        
        print("[Detect] Checking category = %s" % (str(category),))
        
        result = doJob(job_detail, detect_phase=True)
        
        if not result['need_work']:
            print("File `%s` already exist. Skip it." % (result['output_file_fullpath'],))
            continue
        
        input_args.append((job_detail, False))
            
           

print("!!!!!!!!! There are %d jobs needs distribution. " % (len(input_args,)))

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

