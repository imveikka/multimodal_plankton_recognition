#!/bin/bash

PREFIX=./data/FASTVISION-plus/fold

for id in {1..1}
do 

  [[ ! -d ${PREFIX}${id} ]] && \
    python split.py --dataset ./data/FASTVISION-plus --name fold${id} \
    --trainsize 256 --validsize 64 --minsize 320

  # python3 train_multi.py --dataset ${PREFIX}${id} --modelcard ./model_cards/multi/efficientnet_b0_cnn_1_unsupervised_128.yaml
  # python3 train_multi.py --dataset ${PREFIX}${id} --modelcard ./model_cards/multi/efficientnet_b0_cnn_1_supervised_128.yaml
  python3 train_multi.py --dataset ${PREFIX}${id} --modelcard ./model_cards/multi/CLIP_ef0_cnn2.yaml
  python3 train_multi.py --dataset ${PREFIX}${id} --modelcard ./model_cards/multi/SigLIP_ef0_cnn2.yaml

done

