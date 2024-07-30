#!/bin/bash

output_root="output_map_analysis_ERA5"

nproc=1

params=(
    postprocessed AR AR IVT_x
    raw surf_inst sfc msl
)
#    postprocessed AR AR IVT_x
#    postprocessed AR AR IVT_y
#    postprocessed AR AR IWV
#    postprocessed AR AR IVT
#)


#    postprocessed AR AR IVT
#    raw surf_inst sfc u10
#    raw surf_inst sfc u10
#    raw surf_inst sfc v10
#    raw surf_inst sfc msl
#)


nparams=4
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_postraw="${params[$(( i * $nparams + 0 ))]}"
    ECCC_varset="${params[$(( i * $nparams + 1 ))]}"
    ERA5_varset="${params[$(( i * $nparams + 2 ))]}"
    varname="${params[$(( i * $nparams + 3 ))]}"

    echo ":: ECCC_postraw = $ECCC_postraw"
    echo ":: ECCC_varset  = $ECCC_varset"
    echo ":: ERA5_varset = $ERA5_varset"
    echo ":: varname = $varname"

    python3 generate_map_analysis_ERA5.py \
        --year-rng 1998 2017 \
        --ECCC-postraw $ECCC_postraw \
        --ECCC-varset $ECCC_varset \
        --varname $varname \
        --output-root $output_root \
        --nproc $nproc

done
