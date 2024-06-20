#!/bin/bash

# set the configuration file
CONFIG_FILE="vasedeck.txt"

# train NeRF
python nerf-pytorch/run_nerf.py --config $CONFIG_FILE > output.txt
