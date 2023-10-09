#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()

server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": "2023-10-05",
    "expver": "prod",
    "hdate": "2001-10-05",
    "levtype": "sfc",
    "model": "glob",
    "origin": "cwao",
    "param": "151/165/166/228228",
    "step": "24/48/480",
    "stream": "enfh",
    "time": "00:00:00",
    "type": "cf",
    "target": "output"
})
