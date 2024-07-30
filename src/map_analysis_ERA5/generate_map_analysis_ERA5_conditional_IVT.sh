#!/bin/bash



nproc=1

lead_pentads=6
days_per_pentad=5

cond_ERA5_freq=inst
cond_ERA5_varset=postprocessed
cond_varname=IVT
cond_bin_bnds=( 0 250 500 750 1000 2000 )
#lead_pentads=10
#days_per_pentad=3

# Params:

# First two parameters are ECCC data. Fomrat: [raw/postprocessed] [group]
# Second two parameters are ERA5 data. Format: [frequency] [group]
# Last parameter is the variable name shared across ECCC and ERA5

params=(
    postprocessed AR inst AR IVT_y
)


nparams=5

output_root="output-map-analysis-ERA5-conditional-IVT_pentad-${days_per_pentad}_leadpentad-${lead_pentads}"
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

    echo ":: cond_ERA5_freq   = $cond_ERA5_freq"
    echo ":: cond_ERA5_varset = $cond_ERA5_varset"
    echo ":: cond_varname     = $cond_varname"

    python3 generate_map_analysis_ERA5_conditional_IVT.py \
        --year-rng 1998 2017 \
        --ECCC-postraw $ECCC_postraw \
        --ECCC-varset  $ECCC_varset  \
        --ERA5-freq    $ERA5_freq    \
        --ERA5-varset  $ERA5_varset  \
        --varname $varname           \
        --cond-ERA5-freq   $cond_ERA5_freq    \
        --cond-ERA5-varset $cond_ERA5_varset  \
        --cond-varname     $cond_varname      \
        --cond-bin-bnds    "${cond_bin_bnds[@]}" \
        --output-root $output_root   \
        --lead-pentads $lead_pentads \
        --days-per-pentad $days_per_pentad \
        --nproc $nproc

done
