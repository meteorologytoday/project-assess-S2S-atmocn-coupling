#!/bin/bash

output_regional_timeseries_pentad-10-leadpentad-3

params=(
    surf_inst msl  NONE
    AR IVT         NONE
#    AR IVT_y       NONE
#    AR IWV         NONE
#    UVTZ gh 500
#    UVTZ gh 850
)

beg_year=1998
end_year=2017


pentad=5
leadpentad=6

input_root="output_regional_timeseries_pentad-${pentad}-leadpentad-${leadpentad}/${beg_year}-${end_year}"
output_root="fig_regional_timeseries_pentad-${pentad}-leadpentad-${leadpentad}_${beg_year}-${end_year}"

mkdir -p $output_root

nparams=3
for lead_pentad in 3 ; do
#for region_name in "N-PAC" "S-PAC" "T-PAC" "N-ATL" "T-ATL" "S-ATL" "T-IND" "S-IND" "ARC" "SO" ; do
for region_name in "N-PAC" ; do
#for region_name in "NT-IND" "DT-PAC" "NST-PAC" "NW-PAC" "NE-PAC" "N-ATL"; do
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_varset="${params[$(( i * $nparams + 0 ))]}"
    ECCC_varname="${params[$(( i * $nparams + 1 ))]}"
    level="${params[$(( i * $nparams + 2 ))]}"

    if ! [ "$level" = "NONE" ] ; then
        level_str="_level-${level}"
    else
        level_str=""
    fi

    output_file=$output_root/regional_timeseries_region-${region_name}_leadpentad-${lead_pentad}_${ECCC_varset}-${ECCC_varname}${level_str}.png
    


    title="[${region_name}] pentad=${lead_pentad}, ${ECCC_varset}::${ECCC_varname}"

    python3 plot_regional_timeseries.py \
        --input-dir $input_root         \
        --models GEPS5 GEPS6            \
        --region $region_name           \
        --ECCC-varset $ECCC_varset      \
        --ECCC-varname $ECCC_varname    \
        --lead-pentad $lead_pentad      \
        --level $level                  \
        --title "$title"                \
        --output $output_file           \
        --add-datastat \
        --no-display

done
done
done
