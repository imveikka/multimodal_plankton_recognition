#!/bin/bash

DATA=./data/CytoSense/split/

python3 train_image.py --dataset $DATA --modelcard ./model_cards/resnet18.yaml && \
  python3 train_image.py --dataset $DATA --modelcard ./model_cards/resnet50.yaml && \
  python3 train_image.py --dataset $DATA --modelcard ./model_cards/densenet121.yaml && \
  python3 train_image.py --dataset $DATA --modelcard ./model_cards/densenet169.yaml && \
  python3 train_image.py --dataset $DATA --modelcard ./model_cards/efficientnet_b0.yaml && \
  python3 train_image.py --dataset $DATA --modelcard ./model_cards/efficientnet_b1.yaml && \
  python3 train_image.py --dataset $DATA --modelcard ./model_cards/vit_small_16.yaml && \
  python3 train_image.py --dataset $DATA --modelcard ./model_cards/vit_small_32.yaml
