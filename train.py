import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms.v2 as v2

import numpy as np
import pandas as pd
import math
from pathlib import Path
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import sys

sys.path.append('./src')
from src.data import MultiSet, ImageTransforms, ProfileTransform, PairAugmentation
from src.profile_encoder import ProfileTransformer
from src.image_encoder import ImageEncoder
from src.model import BiModal
from src.coordination import DistanceLoss, CLIPLoss, RankLoss

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

annotation_id = 15
max_len = 256

data_path = Path('./data/CytoSense')

image_transforms = ImageTransforms()
signal_transforms = ProfileTransform(max_len=max_len)
pair_augmentation = PairAugmentation()

train_set = MultiSet(annotation_path=data_path / f'train_{annotation_id}.csv', 
                   image_transforms=image_transforms,
                   profile_transform=signal_transforms,
                   pair_augmentation=pair_augmentation)

test_set = MultiSet(annotation_path=data_path / f'test_{annotation_id}.csv', 
                    image_transforms=image_transforms,
                    profile_transform=signal_transforms,
                    pair_augmentation=None)

valid_set = MultiSet(annotation_path=data_path / f'valid_{annotation_id}.csv', 
                    image_transforms=image_transforms,
                    profile_transform=signal_transforms,
                    pair_augmentation=None)

dim_embedding = 256

image_encoder_args = {
    'name': 'efficientnet_b0',
    'pretrained': False,
    'num_classes': 0,
}

profile_encoder_args = {
    'dim_in': 7,
    'dim_out': 64,
    'num_head': 4,
    'num_layers': 6, 
    'dim_feedforward': 2024, 
    'dropout': 0.1, 
    'activation': 'gelu',
    'max_len': max_len,
}

classifier_args = {
    'dim_hidden_layers': (1024,),
    'activation': 'gelu',
    'dropout': 0.1,
}

coordination_args = {
    'method': 'clip',  # distance, clip, or rank
    # 'margin': 5,
    'supervised': True,
    'alpha': .5
}

optim_args = {
    'lr': 1e-4,
    'weight_decay': 1e-10,
}

model = BiModal(
    dim_embed=dim_embedding,
    image_encoder_args=image_encoder_args,
    profile_encoder_args=profile_encoder_args,
    classifier_args=classifier_args,
    coordination_args=coordination_args,
    optim_args=optim_args,
    class_names=train_set.class_names,
)

bs = 32

def multi_collate(batch, model=model):

    image, profile, label = zip(*(sample.values() for sample in batch))

    image_dict = {'image': torch.stack(image)}
    profile_dict = model.profile_encoder.tokenize(profile)
    label_dict = {'label': model.name_to_id(label)}

    return image_dict | profile_dict | label_dict

train_loader = DataLoader(dataset=train_set, batch_size=bs, 
                        shuffle=True, num_workers=8, 
                        drop_last=True, collate_fn=multi_collate)

test_loader = DataLoader(dataset=test_set, batch_size=bs, 
                         num_workers=8, collate_fn=multi_collate)

valid_loader = DataLoader(dataset=valid_set, batch_size=bs, 
                         num_workers=8, drop_last=True, 
                         collate_fn=multi_collate)


logger = TensorBoardLogger(save_dir="logs/", name='foobar')
stopper = EarlyStopping(monitor='Valid/loss_total', min_delta=0.0,
                        patience=10, mode='min')
trainer = Trainer(
    precision='16-mixed',
    min_epochs=10,
    max_epochs=200,
    log_every_n_steps=len(train_loader),
    logger=logger,
    callbacks=[stopper],
)
trainer.fit(model, train_loader, valid_loader)