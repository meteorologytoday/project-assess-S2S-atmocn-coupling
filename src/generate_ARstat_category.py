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
parser.add_argument('--rectifile', action="store_true")
parser.add_argument('--rectifile-threshold', type=float, default=0.0)
parser.add_argument('--days-per-week', type=int, default=7)
parser.add_argument('--number-of-weeks', type=int, default=4)
parser.add_argument('--nproc', type=int, default=1)
args = parser.parse_args()
print(args)


beg_year = args.year_rng[0]
end_year = args.year_rng[1]
number_of_years = end_year - beg_year + 1

number_of_lead_time = 32
days_per_week        = args.days_per_week
number_of_weeks      = args.number_of_weeks
number_of_ensemble   = 4

rectifile_flag = args.rectifile 
rectifile_threshold = args.rectifile_threshold 


print("### rectifile_flag = ", rectifile_flag)
print("### rectifile_threshold = ", rectifile_threshold)


categories = ["CATE0_NOAR", "CATE1_MOAR"]#, "CATE2_HIAR"]
number_of_categories = len(categories)
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



CATE0_NOAR = 0
CATE1_MOAR = 1
CATE2_HIAR = 2

def classify_AR(xa, rectifile=False, rectifile_threshold=0):

    if rectifile == True:
        xa = np.array(xa) > rectifile_threshold
        xa = xa.astype(int)

    #print(xa)    
    sum_x = np.sum(xa)

    category = None

    if sum_x == 0:
        category = CATE0_NOAR
    elif sum_x > 0:
        category = CATE1_MOAR
#
#    elif sum_x <= 2:
#        category = CATE1_MOAR
#    elif sum_x > 2:
#        category = CATE2_HIAR

    else:
        raise Exception("Unable to classify_AR. sum_x = %s" % (str(sum_x),))
    

    return category

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
        output_file = "ECCC-S2S_{model_version:s}_ARoccur-stat-category_{start_months:s}.nc".format(
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

        P_minus_O = dict(
            clim       = np.zeros((number_of_weeks, number_of_samples_per_week, number_of_categories)),
            ECCC       = np.zeros((number_of_weeks, number_of_samples_per_week, number_of_categories)),
        )

        for start_year in range(beg_year, end_year+1):
            for start_month, start_day in start_mds:

                dt = pd.Timestamp(year=start_year, month=start_month, day=start_day)
                ds = open_ARoccur_timeseries_dataset(model_version, args.region, dt)

                da_ARoccur_clim       = ds["ARoccur_clim"] #ARoccur_reanalysis"]
                da_ARoccur_ECCC       = ds["ARoccur_ECCC"]
                da_ARoccur_reanalysis = ds["ARoccur_reanalysis"]
                
                for week in range(number_of_weeks):
                
                    sel_idx = slice(week * days_per_week, (week+1) * days_per_week)

                    category_reanalysis = classify_AR(
                        da_ARoccur_reanalysis.isel(start_time=0, lead_time=sel_idx), 
                        rectifile=rectifile_flag, rectifile_threshold=rectifile_threshold,
                    )

                    category_clim       = classify_AR(
                        da_ARoccur_clim.isel(start_time=0, lead_time=sel_idx),
                        rectifile= rectifile_flag, rectifile_threshold=rectifile_threshold,
                    )

                    category_model      = np.zeros((number_of_ensemble,))
                    for number in range(number_of_ensemble):
                        category_model[number] = classify_AR(da_ARoccur_ECCC.isel(number=number, lead_time=sel_idx), rectifile=rectifile_flag, rectifile_threshold = rectifile_threshold)


                    for category in range(number_of_categories):

                        O       = 1.0 if (category_reanalysis == category) else 0.0
                        P_clim  = 1.0 if (category_clim       == category) else 0.0
                        P_ECCC  = np.mean(category_model == category)

                        P_minus_O["clim"][week, cnt, category] = P_clim - O
                        P_minus_O["ECCC"][week, cnt, category] = P_ECCC - O

                cnt += 1


        #print("check")
        #if np.all(np.abs(P_minus_O["ECCC"][0, :, 0]) == np.abs(P_minus_O["ECCC"][0, :, 1])):
        #    raise Exception("WEIRD P_minus_O!!!!!!!!!!!!!!!")


        # Brier Score
        BS = dict(
            clim = np.zeros((number_of_weeks, number_of_categories)),
            ECCC = np.zeros((number_of_weeks, number_of_categories)),
        )
       

 
        for week in range(number_of_weeks):
            # Compute the Brier Score
            for category in range(number_of_categories):
                for model in ["clim", "ECCC"]:     
                    BS[model][week, category] = np.mean(P_minus_O[model][week, :, category]**2.0)
            
        #if np.all(BS["ECCC"][:, 0] == BS["ECCC"][:, 1]):
        #    raise Exception("WEIRD!!!!!!!!!!!!!!!")


        data_vars = dict()

        for model in ["clim", "ECCC"]:
            
            data_vars["BS_%s" % model] = (
                    ["month_group", "week", "category"],
                    np.reshape(BS[model], (1, number_of_weeks, number_of_categories)),
            )

        ds_new = xr.Dataset(
            data_vars=data_vars,
            
            coords=dict(
                week=(["week",],     np.arange(number_of_weeks)),
                category=(["category",], categories),
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

