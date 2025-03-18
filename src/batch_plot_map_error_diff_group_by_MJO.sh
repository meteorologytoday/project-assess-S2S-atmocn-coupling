#!/bin/bash

source 98_trapkill.sh

batch_cnt_limit=20

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
    UVTZ gh           850
#    AR IVT_y          0

#    surf_inst msl     0
#    AR IWV            0
    surf_avg sst      0
    AR IVT            0
    surf_hf_avg mslhf 0

#    surf_hf_avg msshf 0
#    UVTZ gh           500
#    AR IVT_x          0

)

nparams=3

output_dir=fig_strictMJO_pentad-$days_per_pentad-DJF

#for GEPS6_group in GEPS6sub1 GEPS6sub2 GEPS6 ; do

for GEPS6_group in GEPS6sub1 ; do
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

    output_dir_Emean=${output_dir}/Emean/group-${GEPS6_group}
    output_dir_Estd=${output_dir}/Estd/group-${GEPS6_group}

    mkdir -p $output_dir_Estd
    mkdir -p $output_dir_Emean


    input_dir="output_map_analysis_strictMJO_pentad-${days_per_pentad}-leadpentad-${lead_pentads}"

    for categories in "NonMJO" "P1234" "P5678" "Ambiguous" ; do
        
        categories_str=$( echo "$categories" | sed -e 's/\s\+/-/g' )

        for lead_pentad in $( seq 0 $(( $lead_pentads - 1 )) )  ; do
           
            m_str=$( echo "$months" | sed -r "s/ /,/g" ) 
            output_Emean=$output_dir_Emean/${ECCC_varset}-${varname}${level_str}_${categories_str}_lead-pentad-${lead_pentad}.png
            output_Estd=$output_dir_Estd/${ECCC_varset}-${varname}${level_str}_${categories_str}_lead-pentad-${lead_pentad}.png

            if [ -f "$output_Emean" ] && [ -f "$output_Estd" ] ; then
                echo "Output file $output_Emean and $output_Estd exist. Skip."
            else
                python3 plot_map_prediction_error_diff_group_by_category.py \
                    --input-dir $input_dir \
                    --model-versions GEPS5 $GEPS6_group \
                    --category $categories \
                    --lead-pentad $lead_pentad \
                    --varset $ECCC_varset \
                    --varname $varname \
                    --level $level \
                    --no-display \
                    --output $output_Emean \
                    --output-error $output_Estd \
                    --plot-lat-rng -90 90    \
                    --plot-lon-rng 0 360  &

#                    --plot-lat-rng 0 65    \
#                    --plot-lon-rng 110 250  &


                batch_cnt=$(( $batch_cnt + 1)) 
                if (( $batch_cnt >= $batch_cnt_limit )) ; then
                    echo "Max batch_cnt reached: $batch_cnt"
                    wait
                    batch_cnt=0
                fi
             
            fi
            
        done
    done
done
done
wait


echo "Done."
