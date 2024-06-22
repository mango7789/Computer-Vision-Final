#!/bin/bash

# set the configuration file
CONFIG_FILE="config.txt"

# make log dir
mkdir -p ./logs

# train NeRF and redirect console results to txt
python nerf-pytorch/run_nerf.py --config $CONFIG_FILE >> logs/hhsw.txt
