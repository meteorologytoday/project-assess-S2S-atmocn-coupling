#!/bin/bash

python3 gen_mask.py \
    --test-input ./ECMWF_data/data/raw/CY48R1/ctl/surf_avg/ECMWF-S2S_CY48R1_ctl_surf_avg_2008_01-04.nc \
    --test-varname sst \
    --output region_mask.nc
