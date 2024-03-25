#!/bin/bash

region="30N-40N_130W-120W"

input_dir="analysis_ARoccurence"
output_dir="analysis_ARoccurence_stat/category/$region"

mkdir -p $output_dir

for months in "10" "11" "12" "1" "2" "3" "10 11 12 1 2 3" "12 1 2" ; do
#for months in "10 11 12 1 2 3" ; do
    
    eval "python3 generate_ARstat_category.py \
        --year-rng 1998 2017       \
        --start-time-months $months  \
        --input-dir $input_dir     \
        --output-dir $output_dir \
        --region $region \
        --rectifile \
        --rectifile-threshold 0.1 \
        --days-per-week 5         \
        --number-of-weeks 6       \
        --nproc 1
    " &
done

wait

echo "All done."
