#!/bin/bash

year_beg=1998
year_end=2017



days_per_pentad=10
lead_pentads=3

th=""
for i in $( seq 0 20 ) ; do
    th="$th $(( 25 * $i ))"
done

th_labeled="100 200 300 400 500"
echo $th

params=(
    AR IVT 250 "$th"
)




mkdir -p $output_dir

region_name=westcoast


nparams=4
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_varset="${params[$(( i * $nparams + 0 ))]}"
    varname="${params[$(( i * $nparams + 1 ))]}"
    obs_threshold="${params[$(( i * $nparams + 2 ))]}"
    ROC_thresholds="${params[$(( i * $nparams + 3 ))]}"

    echo ":: ECCC_varset  = $ECCC_varset"
    echo ":: varname = $varname"
    echo ":: obs_thresholds = $obs_threshold"
    echo ":: ROC_thresholds = $ROC_thresholds"

    input_dir="output_regional_timeseries_abs_${region_name}"
    output_dir=fig_ROC_${region_name}_pentad-${days_per_pentad}

    mkdir -p $output_dir

    for region in $( seq 2 ); do
    for months in "11 12 1 2 3"; do

        region_str=$( printf "%04d" $region )
           
        m_str=$( echo "$months" | sed -r "s/ /,/g" ) 
        output=$output_dir/${ECCC_varset}-${varname}_threshold-${obs_threshold}_region-${region_str}_${year_beg}-${year_end}_${m_str}.png


        if [ -f "$output" ] ; then
            echo "Output file $output exists. Skip."
        else
            python3 plot_ROC.py \
                --input-dir $input_dir \
                --region $region_str \
                --model-versions GEPS5 GEPS6sub1 \
                --year-rng $year_beg $year_end \
                --months $months \
                --lead-pentads 0 1 2 3  \
                --varset $ECCC_varset \
                --varname $varname \
                --obs-threshold $obs_threshold \
                --ROC-thresholds $ROC_thresholds \
                --labeled-ROC-thresholds $th_labeled \
                --days-per-pentad $days_per_pentad \
                --level 250 \
                --no-display \
                --output $output & 
        fi
        
    done
    done
done

wait
