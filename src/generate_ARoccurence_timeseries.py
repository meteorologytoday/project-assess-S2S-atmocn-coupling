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

model_versions = ["GEPS5", "GEPS6"]

parser = argparse.ArgumentParser(
                    prog = 'make_ECCC_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)

parser.add_argument('--year-rng', type=int, nargs=2, required=True)
parser.add_argument('--lat-rng', type=float, nargs=2, required=True)
parser.add_argument('--lon-rng', type=float, nargs=2, required=True)
parser.add_argument('--output-root', type=str, required=True)
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)

output_root = args.output_root
number_of_lead_time = 32

# inclusive
year_rng = args.year_rng


ERAinterim_archive_root = "data/ERAinterim"

lat_beg = args.lat_rng[0]
lat_end = args.lat_rng[1]

lon_beg = args.lon_rng[0] % 360.0
lon_end = args.lon_rng[1] % 360.0

latlon_str = "%s-%s_%s-%s" % (
    pretty_latlon.pretty_lat(lat_beg), pretty_latlon.pretty_lat(lat_end),
    pretty_latlon.pretty_lon(lon_beg), pretty_latlon.pretty_lon(lon_end),
)


region_str = latlon_str



def open_ERAinterim_clim_dataset(dt):

    filename = os.path.join(
        ERAinterim_archive_root,
        "stat",
        "ARoccurence",
        "1998-2017_mavg_15days",
        "ERAInterim-clim-daily_{time_str:s}.nc".format(
            time_str = dt.strftime("%m-%d_%H"),
        )
    )
    
    ds = xr.open_dataset(filename)
    
    return ds


def open_ERAinterim_dataset(dt):
    
    filename = os.path.join(
        ERAinterim_archive_root,
        "ARObjects",
        "HMGFSC24_threshold-1998-2017",
        "ARobjs_{time_str:s}.nc".format(
            time_str = dt.strftime("%Y-%m-%d_%H"),
        )
    )
    
    ds = xr.open_dataset(filename).rename(dict(lat="latitude", lon="longitude"))
    
    return ds


def doJob(job_detail, detect_phase=False):

    # phase \in ['detect', 'work']
    result = dict(job_detail=job_detail, status="UNKNOWN", need_work=False, detect_phase=detect_phase, output_file_fullpath=None)

    output_varset = "ARoccur_timeseries"
    try:


        start_time = job_detail['start_time']
        model_version = job_detail['model_version']
        start_time_str = job_detail['start_time'].strftime("%Y-%m-%d")
       
        print("[%s] Start job." % (start_time_str,))
 
        output_dir = os.path.join(
            output_root,
            job_detail['model_version'], 
            output_varset,
            region_str,
        )

        output_file = "ECCC-S2S_{model_version:s}_{varset:s}_{start_time:s}.nc".format(
            model_version = job_detail['model_version'],
            varset        = output_varset,
            start_time    = job_detail['start_time'].strftime("%Y_%m-%d"),
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
        lead_time_vec = [ pd.Timedelta(hours=h) for h in ( 1 + np.arange(number_of_lead_time) ) * 24 ]

        ## First file: ECCC ARoccur
        print("Loading ECCC")
        ds_ECCC = ECCC_tools.open_dataset("postprocessed", "ARObjets", model_version, start_time)
        da_ARoccur_ECCC = xr.where( ds_ECCC["map"] > 0 , 1, 0 ).rename("ARoccur_ECCC")
       
        ## Second file: ERAinterim ARoccur
        print("Loading Reanalysis")
        ds_reanalysis = []
        for i in range(number_of_lead_time):
            start_time_plus_lead_time = start_time + pd.Timedelta(days=1) * i
            ds_reanalysis.append(open_ERAinterim_dataset(start_time_plus_lead_time)["map"])
        
        ds_reanalysis = xr.merge(ds_reanalysis)
        ds_reanalysis = ds_reanalysis.drop_vars(["time",]).rename({"time":"lead_time"}).assign_coords({
            "lead_time" : lead_time_vec,
        })
        #ds_reanalysis.coords["lead_time"].attrs["units"] = "hours"
        ds_reanalysis = ds_reanalysis.expand_dims(dim={"start_time": [start_time,]}, axis=0)
        da_ARoccur_reanalysis = xr.where( ds_reanalysis["map"] > 0 , 1, 0 ).rename("ARoccur_reanalysis")

        ## Third  file: Climatology
        print("Loading Climatology")
        ds_reanalysis = []
        ds_clim = []
        for i in range(number_of_lead_time):
            start_time_plus_lead_time = start_time + pd.Timedelta(days=1) * i
            _tmp = open_ERAinterim_clim_dataset(start_time_plus_lead_time)
            _tmp = _tmp.assign_coords({"time": [start_time_plus_lead_time,]}) # Assign time so that merging won't mess up the order
            ds_clim.append(_tmp)
        
        ds_clim = xr.merge(ds_clim)
        ds_clim = ds_clim.drop_vars(["time",]).rename({"time":"lead_time"}).assign_coords({
            "lead_time" : lead_time_vec,
        })
        #ds_clim.coords["lead_time"].attrs["units"] = "hours"
        ds_clim = ds_clim.expand_dims(dim={"start_time": [start_time,]}, axis=0)
        
        da_ARoccur_clim = ds_clim["ARoccur"].rename("ARoccur_clim")

        print("Reading to do spatial average")
        # Don't wanna merge because they do not necessarily have the same
        # grids.
        data = dict(
            ECCC = da_ARoccur_ECCC,
            reanalysis = da_ARoccur_reanalysis,
            clim = da_ARoccur_clim,
        )
        
        for k, da in data.items():
            #print("Averaging: ", k)
            #print("######## Key = ", k)
            #print(da.coords["latitude"])

            data[k] = da.where(
                ( da.coords["latitude"] >= lat_beg )
                & ( da.coords["latitude"] <= lat_end )
                & ( da.coords["longitude"] >= lon_beg )
                & ( da.coords["longitude"] <= lon_end )
            ).mean(dim=["latitude", "longitude"], skipna=True)


        #data["ECCC"] = data["ECCC"].rename({"latitude":"lat", "longitude":"lon"})
        #data["clim"] = data["clim"].rename({"latitude":"lat2", "longitude":"lon2"})

        print("merging")
        
        #for k, da in data.items():
        #    print(da)

        ds_all = xr.merge(list(data.values()))

        print(ds_all)
        
        #time.sleep(20)
 
        # Make output dir
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
     
        print("Writing to file: %s" % (output_file_fullpath,) )
        ds_all.to_netcdf(
            output_file_fullpath,
            unlimited_dims=["start_time",],
        )

        result['status'] = 'OK'

    except Exception as e:

        print("[%s] Error. Now print stacktrace..." % (start_time_str,))
        import traceback
        traceback.print_exc()


    return result



failed_dates = []
dts_in_year = pd.date_range("2021-01-01", "2021-12-31", inclusive="both")
#dts_in_year = pd.date_range("2021-01-31", "2021-01-31", inclusive="both")
input_args = []
for model_version in model_versions:
    
    print("[MODEL VERSION]: ", model_version)
    
    for dt in dts_in_year:
        
        model_version_date = ECCC_tools.modelVersionReforecastDateToModelVersionDate(model_version, dt)
        
        if model_version_date is None:
            continue
        
        print("The date %s exists on ECMWF database. " % (dt.strftime("%m/%d")))

        for year in range(year_rng[0], year_rng[1]+1):
            
            month = dt.month
            day = dt.day
            start_time = pd.Timestamp(year=year, month=month, day=day)

            print("[Detect] Checking date %s" % (start_time.strftime("%Y-%m-%d"),))

            job_detail = dict(
                model_version = model_version,
                start_time = start_time, 
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

