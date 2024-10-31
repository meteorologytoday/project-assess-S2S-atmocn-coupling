#!/bin/bash

year_beg=1998
year_end=2017



days_per_pentad=5
lead_pentads=6


params=(
    surf_inst msl
    AR IVT_x
    AR IVT_y
    AR IVT
)
#    surf_inst msl

#    surf_inst u10
params=(
    hf_surf_inst mslhf
    hf_surf_inst msshf
)

params=(
    surf_inst msl     0
    AR IVT            0
    UVTZ gh           850
    UVTZ gh           500
    AR IVT_x          0
    AR IVT_y          0
    surf_avg sst      0
)

nparams=3

for GEPS6_group in GEPS6sub1 GEPS6sub2 GEPS6 ; do
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_varset="${params[$(( i * $nparams + 0 ))]}"
    varname="${params[$(( i * $nparams + 1 ))]}"
    level="${params[$(( i * $nparams + 2 ))]}"

    echo ":: ECCC_varset  = $ECCC_varset"
    echo ":: varname = $varname"
    echo ":: level = $level"
    
    if ! [ "$level" = "0" ] ; then
        level_str="-${level}"
    fi 

    output_dir=fig_map_prediction_error_diff_global_pentad-$days_per_pentad-multimonths/group-${GEPS6_group}
    output_error_dir=fig_map_prediction_error_diff_Estd_global_pentad-$days_per_pentad-multimonths/group-${GEPS6_group}

    mkdir -p $output_dir
    mkdir -p $output_error_dir


    input_dir="output_map_analysis_ERA5_pentad-${days_per_pentad}-leadpentad-${lead_pentads}"

    for months in "11 12 1 2 3" ; do
        
        for lead_pentad in $( seq 0 $(( $lead_pentads - 1 )) )  ; do
           
            m_str=$( echo "$months" | sed -r "s/ /,/g" ) 
            output=$output_dir/${ECCC_varset}-${varname}${level_str}_${year_beg}-${year_end}_${m_str}_lead-pentad-${lead_pentad}.png
            output_error=$output_error_dir/${ECCC_varset}-${varname}${level_str}_${year_beg}-${year_end}_${m_str}_lead-pentad-${lead_pentad}.png


            if [ -f "$output" ] && [ -f "$output_error" ] ; then
                echo "Output file $output and $output_error exist. Skip."
            else
                python3 plot_map_prediction_error_diff_group.py \
                    --input-dir $input_dir \
                    --model-versions $GEPS6_group GEPS5 \
                    --year-rng $year_beg $year_end \
                    --months $months \
                    --lead-pentad $lead_pentad \
                    --varset $ECCC_varset \
                    --varname $varname \
                    --level $level \
                    --no-display \
                    --output $output \
                    --output-error $output_error 
            fi
            
        done
    done
done
done
wait
