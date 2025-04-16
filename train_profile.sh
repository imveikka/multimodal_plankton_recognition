#!/bin/bash

PREFIX=./data/FASTVISION-plus/fold

for id in {3..5}
do 

  [[ ! -d ${PREFIX}${id} ]] && \
    python split.py --dataset ./data/FASTVISION-plus --name fold${id} \
    --trainsize 256 --validsize 64 --minsize 320

  # python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/cnn_1.yaml
  # python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/cnn_2.yaml
  # python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/transformer_1.yaml
  # python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/transformer_2.yaml
  python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/lstm_1.yaml
  python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/lstm_2.yaml

done
