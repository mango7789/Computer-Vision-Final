#!/bin/bash

# define the configuration file
CONFIG_FILE="vasedeck.txt"

# test nerf 
python run_nerf.py --config $CONFIG_FILE --render_only
