#!/bin/bash

source 000_setup.sh

python3 src/make_ym_category.py --date-rng 1998-01-01 2017-12-31 --output-dir gendata
python3 ./src/make_MJO_category.py --no-display
python3 ./src/make_nonMJO_category.py --no-display
