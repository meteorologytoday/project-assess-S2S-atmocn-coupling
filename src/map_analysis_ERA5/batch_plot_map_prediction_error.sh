#!/bin/bash

year_beg=1998
year_end=2017

output_dir=fig_map_prediction_error_global

mkdir -p $output_dir

params=(
    surf_inst u10
    surf_inst msl
)

params=(
    AR IVT
    AR IWV
    AR IVT_x
    AR IVT_y
)

params=(
    AR IVT
    AR IVT_x
    AR IVT_y
)

params=(
    surf_inst msl
)



nparams=2
for (( i=0 ; i < $(( ${#params[@]} / $nparams )) ; i++ )); do

    ECCC_varset="${params[$(( i * $nparams + 0 ))]}"
    varname="${params[$(( i * $nparams + 1 ))]}"

    echo ":: ECCC_varset  = $ECCC_varset"
    echo ":: varname = $varname"

    for model_version in GEPS5 GEPS6 ; do

        echo ":::: model_version = $model_version"

        input_dir=output_map_analysis_ERA5/${model_version}

        #for month in $( seq 1 12 ) ; do
        for month in 1 12  ; do
            m_str=$( printf "%02d" $month )
            for lead_pentad in 0 1 2 3 ; do
                
                output=$output_dir/${model_version}_${ECCC_varset}-${varname}_${year_beg}-${year_end}_${m_str}_lead-pentad-${lead_pentad}.png


                if [ -f "$output" ] ; then
                    echo "Output file $output exists. Skip."
                else
                    python3 plot_map_prediction_error.py \
                        --input-dir $input_dir \
                        --model-version $model_version \
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
done

wait
