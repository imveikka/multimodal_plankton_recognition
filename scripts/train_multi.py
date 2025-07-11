import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import argparse
import sys
import json

sys.path.append('../')
from src.data import MultiSet, ImageTransformTrain, ImageTransformTest, \
                     ProfileTransformTrain, ProfileTransformTest, \
                     PairAugmentation
from src.profile_encoder import ProfileTransformer
from src.image_encoder import ImageEncoder
from src.model import MultiModel
from src.coordination import DistanceLoss, CLIPLoss, RankLoss

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

# print(json.dumps(card_dict, indent=4))

precision = card_dict.get('precision', 'highest')
torch.set_float32_matmul_precision(precision)
target_size = card_dict.get('target_size')
bs = card_dict['bs']

data_path = Path(f'{args.dataset}')
dataset = data_path.name

image_transforms_train = ImageTransformTrain(target_size)
signal_transforms_train = ProfileTransformTrain(target_size)
pair_augmentation = PairAugmentation()

image_transforms_test = ImageTransformTest(target_size)
signal_transforms_test = ProfileTransformTest(target_size)

train_set = MultiSet(annotation_path=data_path / f'train.csv', 
                   image_transforms=image_transforms_train,
                   profile_transform=signal_transforms_train,
                   pair_augmentation=pair_augmentation)

test_set = MultiSet(annotation_path=data_path / f'test.csv', 
                    image_transforms=image_transforms_test,
                    profile_transform=signal_transforms_test)

model = MultiModel(
    dim_embed=card_dict['dim_embedding'],
    image_encoder_args=card_dict['image_encoder_args'],
    profile_encoder_args=card_dict['profile_encoder_args'],
    coordination_args=card_dict['coordination_args'],
    optim_args=card_dict['optim_args'],
)

def multi_collate(batch, model=model):

    image, profile, _, image_shape, profile_len = zip(*(sample.values() for sample in batch))

    image = {'image': torch.stack(image)}
    profile = model.profile_encoder.tokenize(profile)
    image_shape = {'image_shape': torch.stack(image_shape)}
    profile_len = {'profile_len': torch.stack(profile_len)}
    buckets = {'buckets': card_dict['buckets']}

    return image | profile | image_shape | profile_len | buckets

train_loader = DataLoader(dataset=train_set, batch_size=bs, 
                        shuffle=True, num_workers=card_dict['num_workers'], 
                        drop_last=True, collate_fn=multi_collate)

valid_loader = DataLoader(dataset=test_set, batch_size=bs, 
                         shuffle=True, num_workers=card_dict['num_workers'],
                         drop_last=True, collate_fn=multi_collate)

name = card.name.split('.')[0] + '_' + '_'.join(str(data_path).split('/')[-2:])
logger = TensorBoardLogger(save_dir="../logs/", name=name)

checkpoint = ModelCheckpoint(
        filename="{epoch}_{valid_loss:.5f}",
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

print(f'Training from model card {args.modelcard}')
trainer.fit(model, train_loader, valid_loader)

