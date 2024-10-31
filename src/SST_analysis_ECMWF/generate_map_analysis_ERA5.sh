#!/bin/bash



nproc=40

lead_pentads=6
days_per_pentad=5


#lead_pentads=10
#days_per_pentad=3

# Params:

# First two parameters are ECMWF data. Fomrat: [raw/postprocessed] [group]
# Second two parameters are ERA5 data. Format: [frequency] [group]
# Last parameter is the variable name shared across ECMWF and ERA5

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
    raw surf_avg  24hr sea_surface_temperature  sea_surface_temperature
)

nparams=5

output_root="output_map_analysis_ERA5_pentad-${days_per_pentad}-leadpentad-${lead_pentads}"
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECMWF_postraw="${params[$(( i * $nparams + 0 ))]}"
    ECMWF_varset="${params[$(( i * $nparams + 1 ))]}"
    ERA5_freq="${params[$(( i * $nparams + 2 ))]}"
    ERA5_varset="${params[$(( i * $nparams + 3 ))]}"
    varname="${params[$(( i * $nparams + 4 ))]}"

    echo ":: ECMWF_postraw = $ECMWF_postraw"
    echo ":: ECMWF_varset  = $ECMWF_varset"
    echo ":: ERA5_freq    = $ERA5_freq"
    echo ":: ERA5_varset  = $ERA5_varset"
    echo ":: varname = $varname"
    
    levels="850 500"


    python3 generate_map_analysis_ERA5.py \
        --year-rng 2004 2022 \
        --ECMWF-postraw $ECMWF_postraw \
        --ECMWF-varset  $ECMWF_varset  \
        --ERA5-freq    $ERA5_freq    \
        --ERA5-varset  $ERA5_varset  \
        --varname $varname           \
        --levels $levels             \
        --output-root $output_root   \
        --lead-pentads $lead_pentads \
        --days-per-pentad $days_per_pentad \
        --nproc $nproc

done
