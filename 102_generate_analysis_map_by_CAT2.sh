#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh

category_file=$gendata_dir/strict_nonMJO_category.csv
date_to_category_file=$gendata_dir/strict_nonMJO_date_to_category.csv
     

nproc=41

days_per_window=5
lead_windows=6

# Params:

# First two parameters are ECCC data. Fomrat: [raw/postprocessed] [group]
# Second two parameters are ERA5 data. Format: [frequency] [group]
# Last parameter is the variable name shared across ECCC and ERA5

params=(
    postprocessed precip daily_acc total_precipitation total_precipitation
    raw surf_inst inst mean_sea_level_pressure mean_sea_level_pressure
    raw surf_avg  daily_mean sea_surface_temperature  sea_surface_temperature
    raw UVTZ inst geopotential geopotential
    raw surf_avg  daily_mean sea_ice_cover sea_ice_cover

    postprocessed AR inst AR IVT
    postprocessed surf_hf_avg daily_mean mean_surface_sensible_heat_flux mean_surface_sensible_heat_flux
    postprocessed surf_hf_avg daily_mean mean_surface_latent_heat_flux mean_surface_latent_heat_flux

#    postprocessed AR inst AR IVT_x
#    postprocessed AR inst AR IVT_y
#    postprocessed AR inst AR IWV
)


nparams=5


#python3 src/make_nonMJO_category.py --no-display

output_root="$gendata_dir/analysis/output_analysis_map_by_CAT2_window-${days_per_window}-leadwindow-${lead_windows}"
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_postraw="${params[$(( i * $nparams + 0 ))]}"
    ECCC_varset="${params[$(( i * $nparams + 1 ))]}"
    ERA5_freq="${params[$(( i * $nparams + 2 ))]}"
    ERA5_varset="${params[$(( i * $nparams + 3 ))]}"
    varname="${params[$(( i * $nparams + 4 ))]}"

    echo ":: ECCC_postraw = $ECCC_postraw"
    echo ":: ECCC_varset  = $ECCC_varset"
    echo ":: ERA5_freq    = $ERA5_freq"
    echo ":: ERA5_varset  = $ERA5_varset"
    echo ":: varname = $varname"
    
    levels="850 500"

    python3 src/generate_analysis_map_by_category.py \
        --category-file $category_file           \
        --date-to-category-file $date_to_category_file \
        --category-name category                 \
        --ECCC-postraw $ECCC_postraw             \
        --ECCC-varset  $ECCC_varset              \
        --ERA5-freq    $ERA5_freq                \
        --ERA5-varset  $ERA5_varset  \
        --varname $varname           \
        --levels $levels             \
        --output-root $output_root   \
        --lead-windows $lead_windows \
        --days-per-window $days_per_window \
        --nproc $nproc
#        --ignored-categories MJO                 \

done

