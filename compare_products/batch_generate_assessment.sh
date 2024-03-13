#!/bin/bash

beg_date=1998-01-01
end_date=2018-03-01

python3 generate_assessment.py \
    --beg-date $beg_date \
    --end-date $end_date \
    --lat-rng -90 90 \
    --lon-rng  0 359 \
    --months 12 1 2 \
    --nproc 5


