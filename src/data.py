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
        self.augmentation = Augmentation()

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        image = torchvision.io.decode_image(self.image_files[index])
        signal = np.loadtxt(self.signal_files[index], delimiter=',', skiprows=1)  
        signal = torch.tensor(signal)

        image = self.image_transforms(image)
        signal = self.signal_transforms(signal)
        y = self.label_encoder.transform([self.labels[index]])

        if self.annotation.startswith('train'):
            image, signal = self.augmentation(image, signal)
        else:
            image = v2.functional.resize(image, 224)

        return image, signal, torch.tensor(y)


class ImageTransforms(nn.Module):

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        bg = find_background_color(image)
        image[0, :25, :] = bg[0]
        image[1, :25, :] = bg[1]
        image[2, :25, :] = bg[2]
        image = square_pad(image, fill=bg)
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        return image


class SignalTransforms(nn.Module):

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal[:, :-1] = signal[:, :-1].add(1).log()
        signal = signal.float()
        return signal


class Augmentation(nn.Module):

    crop = v2.RandomCrop((224, 224))

    def forward(self, image: torch.Tensor, signal: torch.Tensor) -> tuple[torch.Tensor]:
        image = v2.functional.resize(image, 240)
        image = self.crop(image)
        if random.randint(0, 1) == 0:
            image = v2.functional.vertical_flip(image)
        image += (0.01 * torch.randn(image.shape[1:]))
        signal += (0.01 * torch.randn(signal.shape))
        if random.randint(0, 1) == 0:
            image = v2.functional.horizontal_flip(image)
            signal = signal.flip(0)
        return image, signal


def multi_collate(batch: list[tuple[torch.Tensor]], bert: bool = False) -> tuple[torch.Tensor]:

    image, signal, y = zip(*batch)
    lens = map(len, signal)
    batches = len(image)

    image = torch.stack(image)
    signal = nn.utils.rnn.pad_sequence(signal, batch_first=True)
    y = torch.cat(y)
    if bert:
        cls = torch.zeros(batches, 1, signal.shape[-1])
        signal = torch.cat((cls, signal), 1)

    return image, signal, y


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
