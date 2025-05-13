#!/bin/bash

PREFIX=./data/FASTVISION-plus/fold

for id in {2..5}
do 

  python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/cnn_1.yaml
  python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/cnn_2.yaml
  python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/transformer_1.yaml
  python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/transformer_2.yaml
  python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/lstm_1.yaml
  python3 train_profile.py --dataset ${PREFIX}${id} --modelcard ./model_cards/profile/lstm_2.yaml

done
