#!/bin/bash

input_dir="analysis_ARoccurence"
output_dir="analysis_ARoccurence"
region="30N-40N_130W-120W"
python3 generate_ARoccurence_stat.py \
    --year-rng 1998 2017       \
    --start-time-months 1  \
    --input-dir $input_dir     \
    --output-dir $output_dir \
    --region $region \
    --nproc 1
