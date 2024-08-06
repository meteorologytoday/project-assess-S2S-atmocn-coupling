#!/bin/bash

params=(
    surf_inst mean_sea_level_pressure
    AR IWV
    AR IVT
)


params=(
    surf_inst mean_sea_level_pressure
    AR IVT_x
    AR IVT_y
)

params=(
    AR IVT
    AR IWV
    surf_inst mean_sea_level_pressure
)

params=(
    UVTZ geopotential
)




#    surf_inst mean_sea_level_pressure
#    AR IVT
beg_year=1998
end_year=2017

mask_file=test_mask.nc

nparams=2

input_root="output_map_analysis_ERA5_pentad-5-leadpentad-6"
output_root="output_regional_timeseries"

#for region_name in "N-PAC" "S-PAC" "T-PAC" "N-ATL" "T-ATL" "S-ATL" "T-IND" "S-IND" "ARC" "SO" ; do
#for region_name in "NW-PAC" ; do
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_varset="${params[$(( i * $nparams + 0 ))]}"
    ECCC_varname="${params[$(( i * $nparams + 1 ))]}"

    echo ":: ECCC_varset  = $ECCC_varset"
    echo ":: ECCC_varname = $ECCC_varname"
    
    python3 generate_regional_timeseries.py \
        --year-rng $beg_year $end_year   \
        --ECCC-varset  $ECCC_varset      \
        --varname $ECCC_varname          \
        --input-root $input_root         \
        --output-root $output_root       \
        --mask-file $mask_file 

done
#done
