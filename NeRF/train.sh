#!/bin/bash

# set the root dir and configuration file
ROOT_DIR="data/llff/vasedeck"
CONFIG_FILE="vasedeck.txt"

# generate pose file
python LLFF/imgs2poses.py $ROOT_DIR

# train NeRF
python nerf-pytorch/run_nerf.py --config $CONFIG_FILE > output.txt
