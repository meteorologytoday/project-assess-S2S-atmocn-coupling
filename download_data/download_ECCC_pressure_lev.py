#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()

server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": "2023-10-05",
    "expver": "prod",
    "hdate": "2001-10-05",
    "levelist": "200/300/500/700/850/925/1000",
    "levtype": "pl",
    "model": "glob",
    "origin": "cwao",
    "param": "130/131/132/133/156",
    "step": "24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480",
    "stream": "enfh",
    "time": "00:00:00",
    "type": "cf",
    "target": "output"
})
