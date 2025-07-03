#!/bin/bash

PREFIX=./data/FASTVISION-plus/fold

for id in {1..5}
do 

  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/resnet18.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/resnet50.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/densenet121.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/densenet169.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/efficientnet_b0.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/efficientnet_b1.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/vit_small_16.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/vit_small_32.yaml

done


