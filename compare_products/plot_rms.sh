#!/bin/bash

python3 quick_plot_rms.py  --input-file output/ECCC_GEPS5/fcst_error_2015-01.nc --isel-start-time 0 --output fig_rms_GEPS5-2015-01.png --y-rng 0 2500 &
python3 quick_plot_rms.py  --input-file output/ECCC_GEPS6/fcst_error_2015-01.nc --isel-start-time 0 --output fig_rms_GEPS6-2015-01.png --y-rng 0 2500 &


wait
