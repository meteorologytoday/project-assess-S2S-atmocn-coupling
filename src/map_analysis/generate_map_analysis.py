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

parser.add_argument('--start-months', type=int, nargs="+", required=True)
parser.add_argument('--lead-pentads', type=int, default=6)
parser.add_argument('--year-rng', type=int, nargs=2, required=True)
parser.add_argument('--output-root', type=str, required=True)
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)

output_root = args.output_root

# inclusive
year_rng = args.year_rng


ERAinterim_archive_root = "data/ERAinterim"

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


def doJob(job_detail, detect_phase = False):


    result = dict(
        status="UNKNOWN",
        job_detail=job_detail,
    )
 
    
    try: 

        start_ym = job_detail['start_ym']
        dataset = job_detail['dataset']
        varname = job_detail['varname']
      
        start_year  = start_ym.year
        start_month = start_ym.month
       
        output_dir = os.path.join(
            output_root,
            job_detail['model_version'], 
        )

        output_file = "ECCC-S2S_{model_version:s}_{varname:s}_{start_ym:s}.nc".format(
            model_version = job_detail['model_version'],
            varname       = varname,
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
        lead_time_vec = [ pd.Timedelta(hours=h) for h in ( 1 + np.arange(number_of_lead_time) ) * 24 ]

       
        # Decide valid dates
        #test_start_date = pd.Timestamp(year=y, month=m, day=1)
        #test_end_date = test_start_date + pd.offsets.MonthBegin()
        #test_dates = pd.date_range(test_start_date, test_end_date, freq="D", inclusive="left")
        #test_ds = xr.open_dataset(url % ("psl",), decode_times=True)
        #dims = test_ds.dims

        #if dims["M"] != ens_N:
        #    raise Exception("Dimension M != ens_N. M = %d and ens_N = %d" % (dims["M"], ens_N))


        #test_ds = test_ds.sel(S=start_times).isel(X=0, Y=0, M=0, L=0)
        #test_outcome = test_ds["psl"].to_numpy()
        #valid_start_times = start_times[np.isfinite(test_outcome)]
       
         
        # Create dataset
        _tmp = dict()
        print("varname: ", varname)
        _tmp["%s_Emean" % varname]  = (["start_month", "lead_pentad", "latitude", "longitude"], np.zeros((1, len(sel_lat), len(sel_lon))))
        _tmp["%s_E2mean" % varname] = (["start_month", "lead_pentad", "latitude", "longitude"], np.zeros((1, len(sel_lat), len(sel_lon))))
   
        _tmp["start_month"] = (["start_month",], start_month,)
        _tmp["lead_pentad"] = (["lead_pentad",], lead_pentad,)
        
        total_cnt = len(valid_start_times) * len(lead_times) * ens_N
        _tmp["total_cnt"] = (["time",], [total_cnt,],)
        

        output_ds = xr.Dataset(
            data_vars=_tmp,
            coords=dict(
                time=[dt,],
                latitude=(["latitude"], list(sel_lat)),
                longitude=(["longitude"], list(sel_lon)),
            ),
            attrs=dict(
                description="S2S forecast data RMS.",
                total_cnt = total_cnt,
            ),
        )
    
        Emean = np.zeros_like(output_ds["%s_Emean" % (varname,)])
        E2mean = np.zeros_like(output_ds["%s_E2mean" % (varname,)])
                
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

