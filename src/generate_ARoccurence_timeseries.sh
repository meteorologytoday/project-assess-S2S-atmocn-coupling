#!/bin/bash

output_root="analysis_ARoccurence"

python3 generate_ARoccurence_timeseries.py \
    --year-rng 1998 2017 \
    --lat-rng  30 40 \
    --lon-rng -130 -120 \
    --output-root $output_root \
    --nproc 5
