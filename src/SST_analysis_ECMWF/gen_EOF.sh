#!/bin/bash

python3 find_EOF.py                   \
    --model-version CY48R1            \
    --start-year-rng 2004 2022        \
    --start-months 12 1 2             \
    --lead-pentads 5                   \
    --input-dir output_map_analysis_ERA5_pentad-5-leadpentad-6 \
    --output-dir gendata/EOFs         \
    --ECMWF-postraw raw               \
    --ECMWF-varset surf_avg           \
    --varname sea_surface_temperature \
    --modes 15                        \
    --nproc 1                         \
    --mask-file region_mask.nc        \
    --mask-region WWRF                \
    --smooth-pts 5 5 
