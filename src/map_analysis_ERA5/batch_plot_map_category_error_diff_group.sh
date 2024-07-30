#!/bin/bash

year_beg=1998
year_end=2017



days_per_pentad=5
lead_pentads=6

params=(
    surf_inst msl "0,1"
    AR IVT_x "0,1"
    AR IVT_y "0,1"
    AR IVT "0,1"
)

params=(
    AR IVT_y "0,1"
)


output_dir=fig_map_category_prediction_error_diff_global_pentad-$days_per_pentad-multimonth

mkdir -p $output_dir

nparams=3
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_varset="${params[$(( i * $nparams + 0 ))]}"
    varname="${params[$(( i * $nparams + 1 ))]}"
    category="${params[$(( i * $nparams + 2 ))]}"

    echo ":: ECCC_varset  = $ECCC_varset"
    echo ":: varname = $varname"
    echo ":: category = $category"

    input_dir="output_map_category_analysis_ERA5_pentad-${days_per_pentad}-leadpentad-${lead_pentads}"

    for months in "12 1 2" "6 7" ; do
        #m_str=$( printf "%02d" $month )
        m_str=$( echo "$months" | sed -r "s/ /,/g" ) 
        for lead_pentad in $( seq 0 $(( $lead_pentads - 1 )) )  ; do
            
            output=$output_dir/${ECCC_varset}-${varname}_${year_beg}-${year_end}_${m_str}_lead-pentad-${lead_pentad}_category-${category}.png


            if [ -f "$output" ] ; then
                echo "Output file $output exists. Skip."
            else
                python3 plot_map_category_prediction_error_diff_group.py \
                    --input-dir $input_dir \
                    --model-versions GEPS6 GEPS5 \
                    --year-rng $year_beg $year_end \
                    --months $months \
                    --category $category \
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
