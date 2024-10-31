import xarray as xr
import numpy as np
import argparse
from pathlib import Path
import os
import matrix_helper
import dask
import sklearn.decomposition
import ECMWF_tools
from scipy import ndimage

dask.config.set(**{'array.slicing.split_large_chunks': True})


def getFilename(model_version, varset, varname, start_year, start_month):
    
    return Path(model_version) / "ECMWF-S2S_{model_version:s}_{varset:s}::{varname:s}_{start_year:04d}-{start_month:02d}.nc".format(
        model_version = model_version,
        varset = varset,
        varname = varname,
        start_year = start_year,
        start_month = start_month,
    )


def smooth2D(d, size_1, size_2):

    d_smoothed = d.copy()
    
    half_window1 = (size_1 - 1) // 2
    half_window2 = (size_2 - 1) // 2
    N1 = d.shape[0]
    N2 = d.shape[1]
    for j in range(N1):
        rng_1_left = max(0, j-half_window1)
        rng_1_right = min(N1-1, j+half_window1)

        for i in range(N2):
            rng_2_left = max(0, i-half_window2)
            rng_2_right = min(N2-1, i+half_window2)

            if np.isfinite(d_smoothed[j, i]):
                subdata = d[rng_1_left:rng_1_right+1, rng_2_left:rng_2_right+1]
                d_smoothed[j, i] = np.nanmean(subdata)
            else:
                d_smoothed[j, i] = np.nan


    return d_smoothed



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model-version', type=str, required=True)
parser.add_argument('--start-year-rng', type=int, nargs=2, required=True)
parser.add_argument('--start-months', type=int, nargs="+", required=True)
parser.add_argument('--lead-pentads', type=int, help="How many pentads to do.", required=True)
parser.add_argument('--input-dir', type=str, help='Input directory.', required=True)
parser.add_argument('--output-dir', type=str, help='Output directory.', required=True)
parser.add_argument('--ECMWF-postraw', type=str, required=True)
parser.add_argument('--ECMWF-varset', type=str, required=True)
parser.add_argument('--modes', type=int, help="Mask file. If not supplied, take the whole domain.", required=True)
parser.add_argument('--levels', nargs="+", type=int, help="If variable is 3D.", default=None)
parser.add_argument('--varname', type=str, required=True)
parser.add_argument('--mask-file', type=str, required=True)
parser.add_argument('--mask-region', type=str, required=True)
parser.add_argument('--smooth-pts', type=int, nargs=2, default=[1, 1])
parser.add_argument('--nproc', type=int, default=1)

args = parser.parse_args()
print(args)

# inclusive
ECMWF_tools.archive_root = os.path.join("S2S", "ECMWF", "data")

ECMWF_postraw = args.ECMWF_postraw
ECMWF_varset = args.ECMWF_varset
ECMWF_varname_long  = args.varname
ECMWF_varname_short = ECMWF_tools.ECMWF_longshortname_mapping[ECMWF_varname_long]



def loadData(dataset, label, varname, year_rng, pentad_rng):

    all_ds = []

    for y in range(year_rng[0], year_rng[1]+1):

        tp_beg = ptt.TimePentad(year=y, pentad=pentad_rng[0])
        tp_end = ptt.TimePentad(year=y, pentad=pentad_rng[1])

        all_ds.append(data_loader.load_dataset(
            dataset = dataset,
            datatype = "cropped",
            label = label,
            varname = varname,
            tp_beg = tp_beg,
            tp_end = tp_end,
            inclusive = "both",
        ))


    ds = xr.merge(all_ds)
    
    return ds




def work(
    input_dir,
    output_file,
    start_year_rng,
    start_months,
    lead_pentads,
    model_version,
    ECMWF_varname,
    ECMWF_varset,
    ECMWF_postraw,
    mask_file,
    mask_region,
    smooth_pts_y,
    smooth_pts_x,
):
 
    try: 

        data = []
        da_mask = None

        if mask_file != "":    
            print("Loading mask file: ", mask_file)
            da_mask = xr.open_dataset(mask_file)["mask"].sel(region=mask_region)
        
        # Loading data in
        filenames = []
        for start_year in range(start_year_rng[0], start_year_rng[1]+1):
            for start_month in start_months:
                filenames.append(str( input_dir / getFilename(
                    model_version,
                    ECMWF_varset,
                    ECMWF_varname,
                    start_year,
                    start_month,
                )))
        
        ds_all = xr.open_mfdataset(filenames)
       
        lat = ds_all.coords["latitude"] 
        lon = ds_all.coords["longitude"] 

        # Check is mask and datasets are collocated
        if np.any( (da_mask.coords["latitude"].to_numpy() - lat.to_numpy()) > 1e-5 ):

            print("mask: ", da_mask.coords["latitude"].to_numpy() )
            print("data: ", ds_all.coords["latitude"].to_numpy() )

            raise Exception("Latitudes differ.") 
        if np.any( (da_mask.coords["longitude"].to_numpy() - lon.to_numpy()) > 1e-5 ):
            raise Exception("Longitudes differ.")
  
     
        print("Datasets opened.")

        # Determine dimensions
        Ns = len(ds_all.coords["start_ym"])
        Nlat = len(ds_all.coords["latitude"])
        Nlon = len(ds_all.coords["longitude"])
 
        all_EOF = np.zeros( (lead_pentads, args.modes, Nlat, Nlon))
        all_projected_idx = np.zeros( (lead_pentads, args.modes, Ns) )
        all_variance = np.zeros((lead_pentads, args.modes,))
        all_mean = np.zeros((lead_pentads, Nlat, Nlon))
        all_std = np.zeros((lead_pentads, Nlat, Nlon))


        for lead_pentad in range(lead_pentads):
           
            ds = ds_all.sel(lead_pentad=lead_pentad) 
            fulldata = ds["%s_Emean" % ECMWF_varname]

            unsmoothed = fulldata.to_numpy()
            smoothed = np.zeros_like(unsmoothed)
            for s in range(Ns):
                smoothed[s, :, :] = smooth2D(
                    unsmoothed[s, :, :],
                    smooth_pts_y,
                    smooth_pts_x,
                )
            
            fulldata[:, :, :] = smoothed

            fulldata_mean = fulldata.mean(dim="start_ym").to_numpy()
            fulldata_std  = fulldata.std(dim="start_ym").to_numpy()
            fulldata_anom = fulldata.to_numpy() - fulldata_mean
                
            # Make mask and reduction matrix
            mask = np.isfinite(fulldata_mean).astype(int)
            if da_mask is not None:
                mask *= da_mask.to_numpy()

            missing_data_idx = mask == 0
            data_reduction = len(mask) != np.sum(mask)
            d_full = fulldata_anom.reshape((Ns, -1))
            
            if data_reduction:
                print("Data contains NaN. Need to do reduction.")    
            
            M = matrix_helper.constructSubspaceWith(mask.flatten())
            d_reduced = np.zeros((Ns, np.sum(mask)))
         
            print("Reducing the matrix") 
            for s in range(Ns):
                d_reduced[s, :] = M @ d_full[s, :]
            
            pca = sklearn.decomposition.PCA(n_components=args.modes)
            pca.fit(d_reduced)


            EOFs_reduced = pca.components_
            N_components = EOFs_reduced.shape[0]
            EOFs_full = np.zeros((N_components, Nlat, Nlon))
             
            for i in range(N_components):
                EOF_tmp = (M.T @ EOFs_reduced[i, :]).reshape((Nlat, Nlon))
                EOF_tmp[missing_data_idx] = np.nan
                EOFs_full[i, :, :] = EOF_tmp
                
            #           (N_com, features)  (features, Nt) => (N_com, Nt)     
            projected_idx = EOFs_reduced @ d_reduced.T 

            all_EOF[lead_pentad, :, :, :]     = EOFs_full
            all_projected_idx[lead_pentad, :] = projected_idx
            all_variance[lead_pentad, :]      = pca.explained_variance_
            all_mean[lead_pentad, :, :]       = fulldata_mean 
            all_std[lead_pentad, :, :]        = fulldata_std
 
        new_ds = xr.Dataset(
            data_vars = dict(
                EOF = ( ["lead_pentad", "mode", "lat", "lon"], all_EOF ),
                projected_idx = ( ["lead_pentad", "mode", "sample", ], all_projected_idx),
                variance = ( ["lead_pentad", "mode", ], all_variance),
                mean = ( ["lead_pentad", "lat", "lon"], all_mean),
                std = ( ["lead_pentad", "lat", "lon"], all_std),
            ),
            coords = dict(
                lead_pentad = range(lead_pentads),
                lat = lat.to_numpy(),
                lon = lon.to_numpy(),
                sample = np.arange(Ns),
            ),
            attrs = dict(
                varname = ECMWF_varname,
            ),
        )
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        print("Writing output: ", str(output_file))
        new_ds.to_netcdf(output_file)
       
    except Exception as e:
         
        import traceback
        #traceback.print_stack()
        traceback.print_exc()


output_dir = Path(args.output_dir)
output_file = "ECMWF-EOF_{model_version:s}_{varset:s}::{varname:s}_smooth-{smooth_y:d}-{smooth_x:d}_{start_year_rng:s}_{start_months:s}.nc".format(
    model_version = args.model_version,
    varset = ECMWF_varset,
    varname = ECMWF_varname_short,
    start_year_rng = "%d-%d" % (args.start_year_rng[0], args.start_year_rng[1]),
    start_months = "-".join(["%02d" % m for m in args.start_months]),
    smooth_y = args.smooth_pts[0],
    smooth_x = args.smooth_pts[1],
)

print("Target output dir: ", str(output_dir))
print("Target output file: ", str(output_file))

output_file_fullpath = output_dir / output_file

if output_file_fullpath.exists():
    print("File %s already exists. Skip this one." % (str(output_file_fullpath,)))

else:

    print("Doing output file: ", str(output_file)) 
    
    work(
        input_dir = Path(args.input_dir),
        output_file = output_file_fullpath,
        start_year_rng = args.start_year_rng,
        start_months = args.start_months,
        lead_pentads = args.lead_pentads,
        model_version = args.model_version,
        ECMWF_varname = ECMWF_varname_short,
        ECMWF_varset = ECMWF_varset,
        ECMWF_postraw = ECMWF_postraw,
        mask_file = args.mask_file,
        mask_region = args.mask_region,
        smooth_pts_y = args.smooth_pts[0],
        smooth_pts_x = args.smooth_pts[1],
    )

