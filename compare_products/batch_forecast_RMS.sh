#!/bin/bash

beg_date=1998-01-01
end_date=2018-01-01

python3 forecast_RMS.py \
    --beg-date $beg_date \
    --end-date $end_date \
    --lat-rng 30 50 \
    --lon-rng  -180 -130


python3 forecast_RMS.py \
    --beg-date $beg_date \
    --end-date $end_date \
    --lat-rng 30 50 \
    --lon-rng  150 -130


exit

python3 forecast_RMS.py \
    --beg-date $beg_date \
    --end-date $end_date \
    --lat-rng 30 50 \
    --lon-rng -180 -130




exit

# smaller domain
python3 forecast_RMS.py \
    --beg-date $beg_date \
    --end-date $end_date \
    --lat-rng 30 50 \
    --lon-rng -140 -130


# bigger domain
python3 forecast_RMS.py \
    --beg-date $beg_date \
    --end-date $end_date \
    --lat-rng 20 60 \
    --lon-rng 120 -120
