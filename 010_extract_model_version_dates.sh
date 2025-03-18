#!/bin/bash

source 000_setup.sh 

python3 src/extract_model_version_dates.py                \
    --input-file  $data_dir/raw_model_version_dates.txt   \
    --output-file $gendata_dir/model_version_dates.txt 
