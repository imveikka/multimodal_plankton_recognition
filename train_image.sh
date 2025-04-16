#!/bin/bash

PREFIX=./data/FASTVISION-plus/fold

for id in {2..5}
do 

  [[ ! -d ${PREFIX}${id} ]] && \
    python split.py --dataset ./data/FASTVISION-plus --name fold${id} \
    --trainsize 256 --validsize 64 --minsize 320

  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/resnet18.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/resnet50.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/densenet121.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/densenet169.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/efficientnet_b0.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/efficientnet_b1.yaml
  # python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/convnext_tiny.yaml
  # python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/convnext_small.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/vit_small_16.yaml
  python3 train_image.py --dataset ${PREFIX}${id} --modelcard ./model_cards/image/vit_small_32.yaml

done

