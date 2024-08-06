from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import argparse


import traceback
import os
import pretty_latlon
pretty_latlon.default_fmt = "%d"

import shared_info
import ECCC_tools
import ERA5_loader

model_versions = ["GEPS5", "GEPS6"]

parser = argparse.ArgumentParser(
                    prog = 'make_ECCC_AR_objects.py',
                    description = 'Postprocess ECCO data (Mixed-Layer integrated).',
)


# E_5 = E_5 ( time, lead_pentad )
# E_6 = E_6 ( time, lead_pentad )


#parser.add_argument('--start-months', type=int, nargs="+", required=True)
parser.add_argument('--year-rng', type=int, nargs=2, required=True)
parser.add_argument('--input-root', type=str, required=True)
parser.add_argument('--output-root', type=str, required=True)
parser.add_argument('--ECCC-varset', type=str, required=True)
parser.add_argument('--varname', type=str, required=True)
parser.add_argument('--var-threshold', type=float, required=True)
parser.add_argument('--var-threshold-smaller', action="store_true", help="Default is detecting a certain variable exeeds a threshold. If this option is set, then the detection happens when it goes below the threshold.")
parser.add_argument('--mask-file', type=str, required=True)
args = parser.parse_args()
print(args)

output_root = args.output_root
input_root  = args.input_root

# inclusive
year_rng = args.year_rng
ECCC_tools.archive_root = os.path.join(shared_info.ECCC_archive_root)


ECCC_varset = args.ECCC_varset
ECCC_varname_long  = args.varname
ECCC_varname_short = ECCC_tools.ECCC_longshortname_mapping[ECCC_varname_long]


print("Loading mask file: ", args.mask_file)
mask_ds = xr.open_dataset(args.mask_file)

regions = mask_ds.coords["region"].to_numpy()

print("Regions found: ", regions)

region_flags = {
    region : (mask_ds["mask"].sel(region=region) == True).to_numpy() for region in regions
}


def doJob(job_detail, detect_phase = False):

    result = dict(
        status="UNKNOWN",
        job_detail=job_detail,
    )
    
    try: 

        year_rng      = job_detail['year_rng']
        model_version = job_detail['model_version']
        ECCC_varname  = job_detail['ECCC_varname']

        output_dir = os.path.join(
            output_root,
            "{start_y:04d}-{end_y:04d}".format(
                start_y       = year_rng[0],
                end_y         = year_rng[1],
            ),
            model_version, 
        )
        
        output_file = "ECCC-S2S_region_{model_version:s}_{varset:s}::{varname:s}.nc".format(
            model_version = model_version,
            varset        = ECCC_varset,
            varname       = ECCC_varname,
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
 
        start_y = pd.Timestamp(year=year_rng[0], month=1, day=1)
        end_y = pd.Timestamp(year=year_rng[1], month=12, day=1)
        dts = pd.date_range(start_y, end_y, freq="M", inclusive="both")
        
        Emean_varname = "%s_Emean" % ECCC_varname
        E2mean_varname = "%s_E2mean" % ECCC_varname
        Emean = None
        E2mean = None
        total_cnt = None
        level = None

        input_files = []
        for i, dt in enumerate(dts):
            
            input_dir = os.path.join(
                input_root,
                model_version, 
            )

            input_file = "ECCC-S2S_{model_version:s}_{varset:s}::{varname:s}_{start_ym:s}.nc".format(
                model_version = model_version,
                varset        = ECCC_varset,
                varname       = ECCC_varname,
                start_ym      = dt.strftime("%Y-%m"),
            )
   
            input_file_fullpath = os.path.join(input_dir, input_file)
 
            if not os.path.isfile(input_file_fullpath):
                print("File %s does not exist. Skip." % (
                    input_file_fullpath,
                ))
                continue

            print("Load data: ", input_file_fullpath)
            ds = xr.open_dataset(input_file_fullpath).isel(start_ym=0)

            if i == 0:            
                is_var3D = "level" in ds[Emean_varname].coords

                dims = [ len(dts), len(regions), len(ds.coords["lead_pentad"]), ]

                if is_var3D:

                    level = ds.coords["level"]
                    dims += [len(level), ]
     
                Emean  = np.zeros( dims )
                E2mean = np.zeros( dims )
                total_cnt = np.zeros( dims )
                
                Emean[:] = np.nan
                E2mean[:] = np.nan
                
            da_E = ds[Emean_varname]
            da_E2 = ds[E2mean_varname]
            da_cnt = ds["total_cnt"]

            for j, region in enumerate(regions):
                
                region_flag = region_flags[region]
                avg_da_E = da_E.where(region_flag).mean(dim=["latitude", "longitude"], skipna=True, )
                avg_da_E2 = da_E2.where(region_flag).mean(dim=["latitude", "longitude"], skipna=True, )

                _da_cnt = da_cnt.to_numpy()
                selector = [i, j, slice(None),]
               
                if is_var3D:
                    selector += [slice(None), ] 
                    _da_cnt = _da_cnt[:, None]
                 
                selector = tuple(selector)
                #print(selector)
                #print("!!!!!!!!!!!!!!!!!")
                #print(avg_da_E)
                #print(avg_da_E.to_numpy())
                #print(Emean[selector])
                #print(total_cnt[selector])
                #print(_da_cnt)
                #Emean[i, j, :]     = avg_da_E.to_numpy()        
                #E2mean[i, j, :]    = avg_da_E2.to_numpy()        
                #total_cnt[i, j, :] = da_cnt.to_numpy() 
 
                Emean[selector]     = avg_da_E.to_numpy()        
                E2mean[selector]    = avg_da_E2.to_numpy()        
                total_cnt[selector] = _da_cnt
       
        dim_list = ["start_ym", "region", "lead_pentad", ]
        coords = dict(
            start_ym = dts,
            region = regions,
        )

        if is_var3D:
            dim_list += ["level",]
            coords["level"] = level

        output_ds = xr.Dataset(

            data_vars={
                Emean_varname : (dim_list, Emean),
                E2mean_varname : (dim_list, E2mean),
                "total_cnt" : (dim_list, total_cnt),
            },

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


for model_version in model_versions:

    print("Doing model_version: ", model_version)

    job_detail = dict(
        year_rng = args.year_rng,
        model_version = model_version,
        ECCC_varname  = ECCC_varname_short,
    )
    
    result = doJob(job_detail, False)
    
    print(result)

