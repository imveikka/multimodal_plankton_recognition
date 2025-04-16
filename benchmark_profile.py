from sklearn.utils.fixes import tarfile_extractall
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
import numpy as np
from sklearn.metrics import top_k_accuracy_score
import logging
from itertools import chain

sys.path.append('./src')
from src.data import MultiSet, ImageTransforms, ProfileTransform
from src.model import ProfileModel

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# configure logging at the root level of Lightning
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model name")
args = parser.parse_args()

log_path = Path(f'./logs/')
data_path = Path('./data/') 
models = filter(lambda x: x.name.startswith(args.model), log_path.iterdir())
checkpoints = chain(*((m / 'version_0/checkpoints').iterdir() for m in models))

scores = []

for checkpoint in checkpoints:

    print(checkpoint)
    model = ProfileModel.load_from_checkpoint(checkpoint)
    
    max_len = 0 if 'lstm' in args.model else 256 # TODO change this, this is not okay...
    image_transforms = ImageTransforms()
    signal_transforms = ProfileTransform(max_len=max_len)
    
    dataset_path, annotation = str(checkpoint).split('/')[1].split('_')[-2:]
    test_set = MultiSet(annotation_path= data_path / dataset_path / annotation / 'test.csv', 
                        image_transforms=image_transforms,
                        profile_transform=signal_transforms,
                        pair_augmentation=None)

    def multi_collate(batch, model=model):
    
        _, profile, label, _, profile_len = zip(*(sample.values() for sample in batch))
    
        profile = model.tokenize(profile)
        label = {'label': model.name_to_id(label)}
        profile_len = {'profile_len': torch.stack(profile_len)}
    
        return profile | label | profile_len

    test_loader = DataLoader(dataset=test_set, batch_size=128,
                             num_workers=8, collate_fn=multi_collate)
    
    trainer = Trainer(barebones=True)
    
    logits, label = zip(*trainer.predict(model, test_loader))
    logits = torch.cat(logits).numpy()
    label = torch.cat(label).numpy()
    
    top1 = top_k_accuracy_score(label, logits, k=1)
    top5 = top_k_accuracy_score(label, logits, k=5)
    
    scores.append((top1, top5))

m1, m5 = np.mean(scores, 0)
s1, s5 = np.std(scores, 0)

name = args.model
print(f'{name:50} & {m1:.5f}+-{s1:.5f} & {m5:.5f}+-{s5:.5f}')
