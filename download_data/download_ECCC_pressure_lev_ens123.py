#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()

server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": "2023-10-05",
    "expver": "prod",
    "hdate": "2001-10-05",
    "levelist": "200/300/500/700/850/1000",
    "levtype": "pl",
    "model": "glob",
    "number": "1/2/3",
    "origin": "cwao",
    "param": "130/131/132/133/156",
    "step": "24/48",
    "stream": "enfh",
    "time": "00:00:00",
    "type": "pf",
    "target": "output_ens0123.nc",
    'format': 'netcdf',
})

