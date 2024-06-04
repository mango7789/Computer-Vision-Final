#!/bin/bash

model=("CNN" "ViT")
root='./data/'

# define hyper-paramters here
epochs=3
ft_lrs=(5e-5 1e-4 5e-4)
fc_lrs=(1e-3 5e-3 1e-2)
batch_size=(64 128)
beta=1.0

# iterate through all configurations
for model in "${models[@]}"; do
    for ft_lr in "${ft_lrs[@]}"; do
        for fc_lr in "${fc_lrs[@]}"; do
            for batch in "${batch_size[@]}"; do
                # get the output directory
                output="logs/${model}"
                # train with configuration
                python train.py --model $model --epochs $epochs \
                        --ft_lr $ft_lr --fc_lr $fc_lr \
                        --batch $batch --output $output \
                        --root $root --beta $beta
            done
        done
    done
done

