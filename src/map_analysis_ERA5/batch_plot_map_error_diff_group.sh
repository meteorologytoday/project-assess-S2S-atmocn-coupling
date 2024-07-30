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
    surf_inst msl
    surf_avg sst
    AR IVT
    AR IVT_x
    AR IVT_y
)

params=(
    UVTZ gh
)



output_dir=fig_map_prediction_error_diff_global_pentad-$days_per_pentad-multimonths

mkdir -p $output_dir

nparams=2
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_varset="${params[$(( i * $nparams + 0 ))]}"
    varname="${params[$(( i * $nparams + 1 ))]}"

    echo ":: ECCC_varset  = $ECCC_varset"
    echo ":: varname = $varname"

    input_dir="output_map_analysis_ERA5_pentad-${days_per_pentad}-leadpentad-${lead_pentads}"

    for months in "12 1 2" "5 6" ; do
        
        for lead_pentad in $( seq 0 $(( $lead_pentads - 1 )) )  ; do
           
            m_str=$( echo "$months" | sed -r "s/ /,/g" ) 
            output=$output_dir/${ECCC_varset}-${varname}_${year_beg}-${year_end}_${m_str}_lead-pentad-${lead_pentad}.png


            if [ -f "$output" ] ; then
                echo "Output file $output exists. Skip."
            else
                python3 plot_map_prediction_error_diff_group.py \
                    --input-dir $input_dir \
                    --model-versions GEPS6 GEPS5 \
                    --year-rng $year_beg $year_end \
                    --months $months \
                    --lead-pentad $lead_pentad \
                    --varset $ECCC_varset \
                    --varname $varname \
                    --level 500 \
                    --no-display \
                    --output $output 
            fi
            
        done
    done
done

wait
