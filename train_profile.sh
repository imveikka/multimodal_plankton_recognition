#!/bin/bash

DATA=./data/CytoSense/split

python3 train_profile.py --dataset $DATA --modelcard ./model_cards/cnn_1.yaml && \
  python3 train_profile.py --dataset $DATA --modelcard ./model_cards/cnn_2.yaml && \
  python3 train_profile.py --dataset $DATA --modelcard ./model_cards/transformer_1.yaml && \
  python3 train_profile.py --dataset $DATA --modelcard ./model_cards/transformer_2.yaml && \
  python3 train_profile.py --dataset $DATA --modelcard ./model_cards/lstm_1.yaml && \
  python3 train_profile.py --dataset $DATA --modelcard ./model_cards/lstm_2.yaml
