import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import argparse
import sys
import json

sys.path.append('./src')
from src.data import MultiSet, ImageTransforms, ProfileTransform, PairAugmentation
from src.model import ProfileModel

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Location to dataset tables.")
parser.add_argument("-m", "--modelcard", help="Path to model card (yaml file).")
args = parser.parse_args()

card = Path(args.modelcard)

with open(card, 'r') as stream:
    card_dict = yaml.safe_load(stream)

print(json.dumps(card_dict, indent=4))

precision = card_dict.get('precision', 'highest')
torch.set_float32_matmul_precision(precision)
max_len = card_dict.get('max_len')
bs = card_dict['bs']

data_path = Path(f'{args.dataset}')
dataset = data_path.name

image_transforms = ImageTransforms()
signal_transforms = ProfileTransform(max_len=max_len)
pair_augmentation = PairAugmentation()

train_set = MultiSet(annotation_path=data_path / f'{dataset}_train.csv', 
                   image_transforms=image_transforms,
                   profile_transform=signal_transforms,
                   pair_augmentation=pair_augmentation)

test_set = MultiSet(annotation_path=data_path / f'{dataset}_test.csv', 
                    image_transforms=image_transforms,
                    profile_transform=signal_transforms,
                    pair_augmentation=None)

valid_set = MultiSet(annotation_path=data_path / f'{dataset}_valid.csv', 
                    image_transforms=image_transforms,
                    profile_transform=signal_transforms,
                    pair_augmentation=None)

model = ProfileModel(
    profile_encoder_args=card_dict['profile_encoder_args'],
    optim_args=card_dict['optim_args'],
    class_names=train_set.class_names,
)

def multi_collate(batch, model=model):

    _, profile, label, _, profile_len = zip(*(sample.values() for sample in batch))

    profile = model.tokenize(profile)
    label = {'label': model.name_to_id(label)}
    profile_len = {'profile_len': torch.stack(profile_len)}

    return profile | label | profile_len

train_loader = DataLoader(dataset=train_set, batch_size=bs, 
                        shuffle=True, num_workers=4, 
                        drop_last=True, collate_fn=multi_collate)

test_loader = DataLoader(dataset=test_set, batch_size=16, 
                         num_workers=4, collate_fn=multi_collate)

valid_loader = DataLoader(dataset=valid_set, batch_size=32, 
                         num_workers=4, drop_last=True, 
                         collate_fn=multi_collate)

name = card.name.split('.')[0]
logger = TensorBoardLogger(save_dir="logs/", name=name)

checkpoint = ModelCheckpoint(
        filename="{epoch}_{valid_loss:.2f}",
        monitor="valid_loss",
        save_top_k=card_dict.get('save_top_k', 1),
        mode="min"
)
stopper = EarlyStopping(monitor='valid_loss', min_delta=0.0,
                        patience=card_dict['patience'],
                        check_finite=False, mode='min')

trainer = Trainer(
    log_every_n_steps=len(train_loader),
    logger=logger,
    callbacks=[checkpoint, stopper],
    **card_dict['trainer_args']
)

trainer.fit(model, train_loader, valid_loader)
trainer.test(model, test_loader, ckpt_path='best')
