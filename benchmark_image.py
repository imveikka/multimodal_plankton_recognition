from os import wait
from sklearn.utils.fixes import tarfile_extractall
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import logging
from itertools import chain
import matplotlib.pyplot as plt

sys.path.append('./src')
from src.data import MultiSet, ImageTransforms, ProfileTransform
from src.model import ImageModel

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import scienceplots

plt.style.use('science')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 12
})

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
true_all = []
pred_all = []

for checkpoint in checkpoints:

    image_transforms = ImageTransforms()
    signal_transforms = ProfileTransform(max_len=0)

    dataset_path, annotation = str(checkpoint).split('/')[1].split('_')[-2:]
    test_set = MultiSet(annotation_path= data_path / dataset_path / annotation / 'test.csv', 
                        image_transforms=image_transforms,
                        profile_transform=signal_transforms,
                        pair_augmentation=None)


    model = ImageModel.load_from_checkpoint(checkpoint)

    def multi_collate(batch, model=model):

        image, _, label, image_shape, _ = zip(*(sample.values() for sample in batch))

        image = {'image': torch.stack(image)}
        label = {'label': model.name_to_id(label)}
        image_shape = {'image_shape': torch.stack(image_shape)}

        return image| label | image_shape

    test_loader = DataLoader(dataset=test_set, batch_size=128,
                             num_workers=8, collate_fn=multi_collate)

    trainer = Trainer(barebones=True)

    logits, label = zip(*trainer.predict(model, test_loader))
    pred = torch.cat(logits).numpy().argmax(1)
    true = torch.cat(label).numpy()
    
    acc = accuracy_score(true, pred)
    metrics = precision_recall_fscore_support(true, pred, average='macro')
    
    scores.append((acc,) + metrics[:-1])
    true_all.append(true)
    pred_all.append(pred)

m1, m2, m3, m4 = np.mean(scores, 0)
s1, s2, s3, s4 = np.std(scores, 0)

true_all = np.concatenate(true_all)
pred_all = np.concatenate(pred_all)

name = args.model
print(f'{name:50} & {m1:.2%}+-{s1:.2%} & {m2:.2%}+-{s2:.2%} & {m3:.2%}+-{s3:.2%} & {m4:.2%}+-{s4:.2%} \\')
print()
print(classification_report(true_all, pred_all, target_names=model.label_encoder.classes_, digits=4))
print()

disp = ConfusionMatrixDisplay.from_predictions(
    true_all, pred_all, display_labels=model.label_encoder.classes_,
)

fig, ax = plt.subplots(figsize=(10,10))

# Deactivate default colorbar
disp.plot(ax=ax, colorbar=False, xticks_rotation="vertical")

# Adding custom colorbar
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(disp.im_,  cax=cax)

plt.savefig(f'./figures/{name}_cm.pdf')