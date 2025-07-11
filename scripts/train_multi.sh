#!/bin/bash

PREFIX=../data/FASTVISION-plus/fold

for id in {1..5}
do 

  # python3 train_multi.py --dataset ${PREFIX}${id} --modelcard ../model_cards/multi/efficientnet_b0_cnn_2_512_clip.yaml
  # python3 train_multi.py --dataset ${PREFIX}${id} --modelcard ../model_cards/multi/efficientnet_b0_cnn_2_512_siglip.yaml
  python3 train_multi.py --dataset ${PREFIX}${id} --modelcard ../model_cards/multi/vit_t_16_transformer_2_512_clip.yaml
  python3 train_multi.py --dataset ${PREFIX}${id} --modelcard ../model_cards/multi/vit_t_16_transformer_2_512_siglip.yaml

done

