#!/bin/bash

year_beg=1998
year_end=2017

output_dir=fig_map_prediction_error_diff_global

mkdir -p $output_dir

params=(
    surf_inst u10
    surf_inst msl
)

#params=(
#    AR IVT
#    AR IVT_x
#    AR IVT_y
#)



nparams=2
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_varset="${params[$(( i * $nparams + 0 ))]}"
    varname="${params[$(( i * $nparams + 1 ))]}"

    echo ":: ECCC_varset  = $ECCC_varset"
    echo ":: varname = $varname"

    input_dir=output_map_analysis

    for month in 12 1  ; do
        m_str=$( printf "%02d" $month )
        for lead_pentad in 0 1 2 ; do
            
            output=$output_dir/${ECCC_varset}-${varname}_${year_beg}-${year_end}_${m_str}_lead-pentad-${lead_pentad}.png


            if [ -f "$output" ] ; then
                echo "Output file $output exists. Skip."
            else
                python3 plot_map_prediction_error_diff_group.py \
                    --input-dir $input_dir \
                    --model-versions GEPS5 GEPS6 \
                    --year-rng $year_beg $year_end \
                    --month $month \
                    --lead-pentad $lead_pentad \
                    --varset $ECCC_varset \
                    --varname $varname \
                    --no-display \
                    --output $output 
            fi
            
        done
    done
done

wait
