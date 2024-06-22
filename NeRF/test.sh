#!/bin/bash

# define the configuration file
CONFIG_FILE="config.txt"

# test nerf 
python nerf-pytorch/run_nerf.py --config $CONFIG_FILE --render_only
