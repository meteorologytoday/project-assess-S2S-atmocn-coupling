#!/bin/bash
py=python3
sh=bash


fig_dir=figures

data_dir=./data
gendata_dir=./gendata


export PYTHONPATH=$( realpath "$( pwd )/lib"):$PYTHONPATH

mkdir -p $gendata_dir
mkdir -p $fig_dir



