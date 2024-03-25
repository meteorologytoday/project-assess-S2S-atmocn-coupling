#!/bin/bash

region=30N-40N_130W-120W
input_dir="analysis_ARoccurence_stat/$region"

fig_dir=fig_ARoccur_stat

mkdir -p $fig_dir


#for months in "10" "11" "12" "01" "02" "03" "12-01-02"; do
for months in "10-11-12-01-02-03"; do

    input_file1="$input_dir/ECCC-S2S_GEPS5_ARoccur-stat_${months}.nc"
    input_file2="$input_dir/ECCC-S2S_GEPS6_ARoccur-stat_${months}.nc"

    python3 plot_ARoccurence_stat.py \
        --input-files $input_file1 $input_file2 \
        --dataset-names GEPS5 GEPS6 \
        --output $fig_dir/ARoccur-stat_${months}.png \
        --title "Month: $months" \
        --no-display

done
