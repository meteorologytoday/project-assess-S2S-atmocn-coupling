#!/bin/bash

category_file=strictMJO_category.csv
date_to_category_file=strictMJO_date_to_category.csv
     

nproc=40

days_per_pentad=5
lead_pentads=6

# Params:

# First two parameters are ECCC data. Fomrat: [raw/postprocessed] [group]
# Second two parameters are ERA5 data. Format: [frequency] [group]
# Last parameter is the variable name shared across ECCC and ERA5

params=(
    raw surf_inst inst mean_sea_level_pressure mean_sea_level_pressure
    postprocessed AR inst AR IVT
    postprocessed AR inst AR IVT_x
    postprocessed AR inst AR IVT_y
    postprocessed AR inst AR IWV
    postprocessed hf_surf_inst inst mean_surface_sensible_heat_flux mean_surface_sensible_heat_flux
    postprocessed hf_surf_inst inst mean_surface_latent_heat_flux mean_surface_latent_heat_flux
)

params=(
    raw surf_inst inst mean_sea_level_pressure mean_sea_level_pressure
    postprocessed AR inst AR IVT
    postprocessed surf_hf_avg daily_mean mean_surface_sensible_heat_flux mean_surface_sensible_heat_flux
    postprocessed surf_hf_avg daily_mean mean_surface_latent_heat_flux mean_surface_latent_heat_flux
    raw surf_avg  daily_mean sea_surface_temperature  sea_surface_temperature
    postprocessed AR inst AR IVT_x
    postprocessed AR inst AR IVT_y
    postprocessed AR inst AR IWV
    raw UVTZ inst geopotential geopotential
)

params=(
    raw surf_avg  daily_mean sea_surface_temperature  sea_surface_temperature
    raw UVTZ inst geopotential geopotential
    raw surf_inst inst mean_sea_level_pressure mean_sea_level_pressure
    postprocessed AR inst AR IVT
    postprocessed surf_hf_avg daily_mean mean_surface_sensible_heat_flux mean_surface_sensible_heat_flux
    postprocessed surf_hf_avg daily_mean mean_surface_latent_heat_flux mean_surface_latent_heat_flux

    postprocessed AR inst AR IVT_x
    postprocessed AR inst AR IVT_y
    postprocessed AR inst AR IWV
)


nparams=5


#python3 make_MJO_category.py
python3 make_strict_MJO_category.py --no-display

output_root="output_map_analysis_strictMJO_pentad-${days_per_pentad}-leadpentad-${lead_pentads}"
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


    python3 generate_map_analysis_by_category.py \
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
        --lead-pentads $lead_pentads \
        --days-per-pentad $days_per_pentad \
        --nproc $nproc

done
