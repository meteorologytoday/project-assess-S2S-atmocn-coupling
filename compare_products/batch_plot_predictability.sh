#!/bin/bash


data_dir=output_fcst_error_10N-60N_120E-120W

python3 plot_predictability.py \
    --input-dirs $data_dir/ECCC_GEPS5 $data_dir/ECCC_GEPS6 \
    --year-rng 1998-10 2017-05
