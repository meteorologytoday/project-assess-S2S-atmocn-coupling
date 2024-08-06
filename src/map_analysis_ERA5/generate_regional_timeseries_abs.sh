#!/bin/bash



nproc=1

# Params:

# First two parameters are ECCC data. Fomrat: [raw/postprocessed] [group]
# Second two parameters are ERA5 data. Format: [frequency] [group]
# Last parameter is the variable name shared across ECCC and ERA5

params=(
    postprocessed AR inst AR IVT
)

nparams=5

output_root="output_regional_timeseries_abs"
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


    python3 generate_regional_ROC2.py \
        --year-rng 1998 2017 \
        --mask-file mask_westcoast.nc \
        --ECCC-postraw $ECCC_postraw \
        --ECCC-varset  $ECCC_varset  \
        --ERA5-freq    $ERA5_freq    \
        --ERA5-varset  $ERA5_varset  \
        --varname $varname           \
        --levels $levels             \
        --output-root $output_root   \
        --nproc $nproc

done
