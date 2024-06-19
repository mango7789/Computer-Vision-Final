#!/bin/bash

# set evaluation configurations
export ROOT_DIR="data/llff/dragon/" 
export IMG_W="504" 
export IMG_H="378"  
export SCENE="dragon"  
export CKPT_PATH="/content/epoch.40.ckpt"

# Run the evaluation script
python nerf_pl/eval.py \
   --root_dir "$ROOT_DIR" \
   --dataset_name llff --scene_name "$SCENE" \
   --img_wh $IMG_W $IMG_H --N_importance 64 --ckpt_path "$CKPT_PATH"
