#!/bin/bash

TRAIN_SCRIPT="train.py"
mkdir -p ./model


# train the SimCLR model
echo "Training the SimCLR model..."
python $TRAIN_SCRIPT simclr --save
echo "SimCLR model training completed and saved."


# train the ResNet-18 model without pretraining
echo "Training the ResNet-18 model..."
python $TRAIN_SCRIPT resnet --save
echo "ResNet-18 model training completed and saved."


# extract and train the linear classifier for each model type
MODELS=("simclr.pth" "resnet_with_pretrain.pth" "resnet_no_pretrain.pth")
TYPES=("self_supervise" "supervise_with_pretrain" "supervise_no_pretrain")


for i in ${!MODELS[@]}; do
    MODEL=${MODELS[$i]}
    TYPE=${TYPES[$i]}

    echo "Extracting and training the linear classifier for model $MODEL with type $TYPE..."
    python $TRAIN_SCRIPT linear \
        --model "./model/$MODEL" \
        --type $TYPE \
        --save
    echo "Linear classifier training for model $MODEL with type $TYPE completed."
done


echo "All tasks completed."
