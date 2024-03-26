#!/bin/bash

year_beg=1998
year_end=2017

output_dir=fig_map_prediction_error_global

mkdir -p $output_dir

for dataset in GEPS6 GEPS5; do

    input_dir=output_fcst_error_90S-90N_0E-1W/ECCC_${dataset}

    #for month in $( seq 1 12 ) ; do
    for month in 1 12  ; do
        m_str=$( printf "%02d" $month )
        for pentad in 0 1 2 3 ; do
            
            output=$output_dir/${dataset}_${year_beg}-${year_end}_${m_str}_pentad-${pentad}.png


            if [ -f "$output" ] ; then
                echo "Output file $output exists. Skip."
            else
                python3 plot_map_prediction_error.py \
                    --input-dir $input_dir \
                    --dataset-name $dataset \
                    --year-rng $year_beg $year_end \
                    --month $month \
                    --pentad $pentad \
                    --no-display \
                    --output $output &
            fi
            
        done

        sleep 1
    done
done


wait
