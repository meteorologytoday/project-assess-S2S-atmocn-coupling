print("Load libraries...")

from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import argparse
import shared_info


import traceback
import os
import pretty_latlon
pretty_latlon.default_fmt = "%d"

import ECCC_tools
import ERA5_loader

print("Done.")

parser = argparse.ArgumentParser(
                    prog = 'make_ECCC_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

#parser.add_argument('--start-months', type=int, nargs="+", required=True)
parser.add_argument('--year-rng', type=int, nargs=2, required=True)
parser.add_argument('--model-versions', type=str, nargs="+", required=True)
parser.add_argument('--output-root', type=str, required=True)
parser.add_argument('--ECCC-postraw', type=str, required=True)
parser.add_argument('--ECCC-varset', type=str, required=True)
parser.add_argument('--ERA5-varset', type=str, required=True)
parser.add_argument('--mask-file', type=str, required=True)
parser.add_argument('--ERA5-freq', type=str, required=True)
parser.add_argument('--levels', nargs="+", type=int, help="If variable is 3D.", default=None)
parser.add_argument('--varname', type=str, required=True)
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)

output_root = args.output_root

mask_ds = xr.open_dataset(args.mask_file)
regions = mask_ds.coords["region"].to_numpy()

print("Regions found: ", regions)
region_flags = {
    region : mask_ds["mask"].sel(region=region) == True for region in regions
}


# inclusive
year_rng = args.year_rng

ECCC_tools.archive_root = os.path.join(shared_info.ECCC_archive_root)


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

        output_file = "ECCC-S2S_regional_{model_version:s}_{varset:s}::{varname:s}_{start_ym:s}.nc".format(
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

        variable3D = "level" in aux_ds[ECCC_varname].dims
        do_level_sel = variable3D and args.levels is not None
            
        dim_ECCC = (
            len(start_times),
            len(aux_ds.coords["lead_time"]),
            len(aux_ds.coords["number"]),
            len(regions),
        )

        dim_ref = (
            len(start_times),
            len(aux_ds.coords["lead_time"]),
            len(regions),
        )


        if variable3D:
            if args.levels is None:
                nlev = aux_ds.dims["level"]

            else:
                nlev = len(args.levels)

            dim_ECCC = (*dim_ECCC, nlev, )
            dim_ref  = (*dim_ref,  nlev, )
            
        var_ECCC_mean  = np.zeros(dim_ECCC)
        var_ref_mean   = np.zeros(dim_ref)

        # This variable is used to adjust the time specifiction
        # between ECCC and ERA5.
        # For example, sst is a daily average, the first lead time is 12 hours (12Z) whereas I 
        # stored the average time as 00Z.
        
        if ERA5_freq == "daily_mean": 
            ERA5_time_adjustment = - pd.Timedelta(hours=12)
        else:
            ERA5_time_adjustment = - pd.Timedelta(days=1)
         
        for st, start_time in enumerate(start_times):

            print("start_time: ", start_time)
        
            ds_ECCC = ECCC_tools.open_dataset(ECCC_postraw, ECCC_varset, model_version, start_time).isel(start_time=0)
            
            for lt, lead_time in enumerate(ds_ECCC.coords["lead_time"].to_numpy()):
            
                _ds_ECCC = ds_ECCC[ECCC_varname].isel(lead_time=lt)
                
                if do_level_sel:
                    _ds_ECCC = _ds_ECCC.sel(level=args.levels)

                start_time_plus_lead_time = start_time + lead_time
                    
                ref_data = ERA5_loader.open_dataset_ERA5(
                    start_time + lead_time + ERA5_time_adjustment,
                    ERA5_freq,
                    ERA5_varset,
                )[ERA5_varname].isel(time=0)
                
                if do_level_sel:
                    ref_data = ref_data.sel(level=args.levels)
                
                    
                if args.varname == "geopotential":
                    ref_data /= 9.81 

                # Interpolation
                #print("ref_data: ", ref_data.coords["latitude"].to_numpy())
                #print("_ds_ECCC: ", _ds_ECCC.coords["latitude"].to_numpy())
                ref_data = ref_data.interp(
                    coords=dict(
                        latitude  = _ds_ECCC.coords["latitude"],
                        longitude = _ds_ECCC.coords["longitude"],
                    ),
                )
                
                for j, region in enumerate(regions):
            
                    region_flag = region_flags[region]
                    avg_ECCC_da = _ds_ECCC.where(region_flag).mean(dim=["latitude", "longitude"], skipna=True, )
                    avg_ref_da  = ref_data.where(region_flag).mean(dim=["latitude", "longitude"], skipna=True, )

                    #print("avg_ECCC_da: " , avg_ECCC_da) 
                    #print("avg_ref_da: " , avg_ref_da) 
                    var_ECCC_mean[st, lt, :, j] = avg_ECCC_da.to_numpy() 
                    var_ref_mean[st, lt, j] = avg_ref_da.to_numpy() 

        # prepping for output
        coords=dict(
            start_time=start_times,
            lead_time=aux_ds.coords["lead_time"],
            number=aux_ds.coords["number"],
            region=regions,
        )

            
        dim_list_ECCC  = ["start_time", "lead_time", "number", "region"] 
        dim_list_ref   = ["start_time", "lead_time", "region"] 
        if variable3D:
            dim_list_ECCC   += ["level",]
            dim_list_ref    += ["level,"]
            if args.levels is None:
                coords["level"] = aux_ds.coords["level"]
            else:
                coords["level"] = (["level",], np.array(args.levels, dtype=float))


        _tmp = dict()

        _tmp["ECCC_%s" % (ECCC_varname, )]  = (dim_list_ECCC, var_ECCC_mean)
        _tmp["ERA5_%s" % (ECCC_varname, )]  = (dim_list_ref,  var_ref_mean)

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


start_yms = pd.date_range(
    pd.Timestamp(year=year_rng[0], month=1,  day=1),
    pd.Timestamp(year=year_rng[1]+1, month=1, day=1),
    freq="M",
    inclusive="both",
)

for model_version in args.model_versions:
    
    print("[MODEL VERSION]: ", model_version)
    
    for start_ym in start_yms:

        #if start_ym.month not in [4,]:
        #    print("For now skip doing this month: ", str(start_ym))
        #    continue
 
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

