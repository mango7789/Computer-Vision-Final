#!/bin/bash

# set training configurations
export ROOT_DIR="data/llff/dragon/"  
export IMG_W="504"  
export IMG_H="378"  
export NUM_EPOCHS="30"  
export EXP="exp"  

# get the `poses_bounds.npy` file
python LLFF/imgs2poses.py $ROOT_DIR

# train NeRF on the dataset
python nerf_pl/train.py \
   --dataset_name llff \
   --root_dir "$ROOT_DIR" \
   --N_importance 64 --img_wh $IMG_W $IMG_H \
   --spheric --use_disp \
   --num_epochs $NUM_EPOCHS --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler cosine \
   --exp_name $EXP

