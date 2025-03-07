import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class MultiSet(Dataset):

    def __init__(self, data_path: Path, annotation: str) -> None:
        super().__init__()

        self.data_path = data_path
        self.annotation = annotation
        self.table = pd.read_csv(data_path / f'{annotation}.csv')
        self.label_encoder = LabelEncoder().fit(self.table.class_name)

        self.X = self.table.X.tolist()
        self.image_files = [data_path / f'images/{x}.jpg' for x in self.X]
        self.signal_files = [data_path / f'others/{x}.csv' for x in self.X]
        self.labels = self.table.class_name.to_list()
        
        self.image_transforms = ImageTransforms()
        self.signal_transforms = SignalTransforms()

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        image = torchvision.io.decode_image(self.image_files[index])
        signal = np.loadtxt(self.signal_files[index], delimiter=',', skiprows=1)  
        signal = torch.tensor(signal)
        y = self.label_encoder.transform([self.labels[index]])
        return self.image_transforms(image), self.signal_transforms(signal), torch.tensor(y)


class ImageTransforms(nn.Module):

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image[:, 20:, :]
        bg = find_background_color(image)
        image = square_pad(image, fill=bg)
        image = v2.functional.resize(image, 224)
        return v2.functional.to_dtype(image, dtype=torch.float32, scale=True)


class SignalTransforms(nn.Module):

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal[:, :-1] = signal[:, :-1].add(1).log()
        signal = signal.float()
        return signal


def collate_bert(batch: list[tuple[torch.Tensor]]) -> tuple[torch.Tensor]:

    image, signal, y = zip(*batch)
    lens = map(len, signal)
    batches = len(image)

    image = torch.stack(image)
    signal = nn.utils.rnn.pad_sequence(signal, batch_first=True)
    cls = torch.zeros(batches, 1, signal.shape[-1])
    signal = torch.cat((cls, signal), 1)
    y = torch.cat(y)

    time = torch.zeros(signal.shape[:2]).long()
    for i, l in enumerate(lens):
        time[i, :l+1] = torch.arange(1, l+2)

    return image, signal, time, y


def find_background_color(image: torch.Tensor, p: int = 2) -> list[int]:

    """
    Finds the background color from image. 
    Based on mode from image rim of thickness p (pixels).
    Can be then used to fill background.
    """

    return torch.cat(
        [
            image[:, :p, :].reshape(3, -1),
            image[:, -p:, :].reshape(3, -1),
            image[:, :, :p].reshape(3, -1),
            image[:, :, -p:].reshape(3, -1),
        ], 
        dim=1
    ).mode(1).values.numpy()


def square_pad(image: torch.Tensor, fill: int | list[int] = 0) -> torch.Tensor:

    """
    Pads an image to square shape.
    """

    c, h, w = image.shape
    d = h - w
    if d < 0: padding = (0, -d // 2 + (-d % 2), 0, -d // 2)
    elif d > 0: padding = (d // 2 + (d % 2), 0, d // 2, 0)
    else: padding = (0, 0)
    return v2.functional.pad(image, padding=padding, fill=fill)


def get_class_images(main_folder_path, class_name):
    folder_path = os.path.join(main_folder_path, class_name)
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    return image_files
