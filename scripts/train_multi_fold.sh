#!/bin/bash

FOLD=$1

if [[ ! -d "$FOLD" ]]; then
  echo "Directory doesn't exits!"
else
    # python3 train_multi.py --dataset ${FOLD} --modelcard ../model_cards/multi/efficientnet_b0_cnn_2_512_clip.yaml
    python3 train_multi.py --dataset ${FOLD} --modelcard ../model_cards/multi/efficientnet_b0_cnn_2_512_siglip.yaml
    # python3 train_multi.py --dataset ${FOLD} --modelcard ../model_cards/multi/vit_s_16_transformer_2_512_clip.yaml
    python3 train_multi.py --dataset ${FOLD} --modelcard ../model_cards/multi/vit_t_16_transformer_2_512_siglip.yaml
fi

