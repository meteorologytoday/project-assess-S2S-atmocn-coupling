#!/bin/bash

params=(
    AR IVT         NONE
    AR IVT_y       NONE
    AR IWV     
    surf_inst msl  NONE
)

params=(
    AR IWV         0
    AR IVT         0
    surf_inst msl  0
    UVTZ gh        850
)


beg_year=1998
end_year=2017



pentad=5
leadpentad=6

input_root="output_regional_timeseries_pentad-${pentad}-leadpentad-${leadpentad}/${beg_year}-${end_year}"
output_root="fig_regional_timeseries_pentad-${pentad}-leadpentad-${leadpentad}_${beg_year}-${end_year}"


mkdir -p $output_root

nparams=3
for region_name in "N-PAC" ; do # "NE-PAC" "NW-PAC" ; do # "S-PAC" "T-PAC" "N-ATL" "T-ATL" "S-ATL" "T-IND" "S-IND" "ARC" "SO" ; do
#for region_name in "T-PAC" "T-ATL" "NW-PAC" "NE-PAC" "N-ATL"; do
#for region_name in "N-ATL" "NT-PAC" "NT-IND" "NDT-PAC" "NST-PAC" "NW-PAC" "NE-PAC" "NT-ATL"; do
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_varset="${params[$(( i * $nparams + 0 ))]}"
    ECCC_varname="${params[$(( i * $nparams + 1 ))]}"
    level="${params[$(( i * $nparams + 2 ))]}"

    if ! [ "$level" = "0" ] ; then
        level_str="_level-${level}"
    else
        level_str=""
    fi


    output_file=$output_root/diff_regional_timeseries_region-${region_name}_${ECCC_varset}-${ECCC_varname}${level_str}.png
    
    if [ -f "$output_file" ] ; then

        echo "File exists: $output_file"

    else
        python3 plot_regional_timeseries_diff.py \
            --input-dir $input_root         \
            --models GEPS5 GEPS6sub1        \
            --region $region_name           \
            --ECCC-varset $ECCC_varset       \
            --ECCC-varname $ECCC_varname     \
            --lead-pentads 0 1 2 3           \
            --month-span 6                   \
            --output $output_file            \
            --level $level                   \
            --percentage                     \
            --days-per-pentad $pentad        \
            --add-datastat \
            --no-display

    fi

done
done
