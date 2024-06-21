#!/bin/bash

# set the configuration file
CONFIG_FILE="config.txt"

# make log dir
mkdir -p ./logs

# train NeRF
python nerf-pytorch/run_nerf.py --config $CONFIG_FILE >> logs/hhsw.txt
