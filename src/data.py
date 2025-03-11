import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2

import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class MultiSet(Dataset):

    def __init__(self, data_path: Path, annotation_file: str,
                 image_transforms: object, image_size: int | tuple[int],
                 signal_transforms: object) -> None:
        super().__init__()

        self.data_path = data_path
        self.annotation_file = annotation_file
        self.table = pd.read_csv(data_path / annotation_file)
        self.label_encoder = LabelEncoder().fit(self.table.class_name)

        self.X = self.table.X.tolist()
        self.image_files = [data_path / f'images/{x}.jpg' for x in self.X]
        self.signal_files = [data_path / f'others/{x}.csv' for x in self.X]
        self.labels = self.table.class_name.to_list()
        
        self.image_transforms = image_transforms
        self.signal_transforms = signal_transforms
        self.augmentation = Augmentation(image_size=image_size)
        self.image_size = image_size

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        image = torchvision.io.decode_image(self.image_files[index])
        signal = np.loadtxt(self.signal_files[index], delimiter=',', skiprows=1)  
        signal = torch.tensor(signal)

        image = self.image_transforms(image)
        signal = self.signal_transforms(signal)
        y = self.label_encoder.transform([self.labels[index]])

        if self.annotation_file.startswith('train'):
            image, signal = self.augmentation(image, signal)
        else:
            image = v2.functional.resize(image, self.image_shape)

        return image, signal, torch.tensor(y)


class ImageTransforms(object):

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        bg = find_background_color(image)
        image[0, :25, :] = bg[0]
        image[1, :25, :] = bg[1]
        image[2, :25, :] = bg[2]
        image = square_pad(image, fill=bg)
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        return image


class SignalTransforms(object):

    def __init__(self, max_len: int = None):
        self.min = torch.tensor([0, 0, 0, 0, 0, 0, -1])
        self.max = torch.tensor([14850, 7360, 408, 7360, 7488, 7488, 1])
        self.diff = self.max - self.min
        self.max_len = max_len

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        signal = (signal - self.min) / self.diff
        signal = signal.float()
        if self.max_len:
            signal = constrait_len(signal, max_len=self.max_len)
        return signal


class Augmentation(object):

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.expanded = tuple(math.floor(1.1 * s) for s in self.image_size)
        self.crop = v2.RandomCrop(self.image_size)

    def __call__(self, image: torch.Tensor, signal: torch.Tensor) -> tuple[torch.Tensor]:
        image = v2.functional.resize(image, self.expanded)
        image = self.crop(image)
        if random.randint(0, 1) == 0:
            image = v2.functional.vertical_flip(image)
        image += (5e-3 * torch.randn(image.shape[1:]))
        signal += (5e-3 * torch.randn(signal.shape))
        if random.randint(0, 1) == 0:
            image = v2.functional.horizontal_flip(image)
            signal = signal.flip(0)
        return image, signal


def multi_collate(batch: list[tuple[torch.Tensor]]) -> tuple[torch.Tensor]:

    image, signal, y = zip(*batch)
    lens = map(len, signal)
    batches = len(image)

    image = torch.stack(image)
    signal = nn.utils.rnn.pad_sequence(signal, batch_first=True)
    y = torch.cat(y)

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

def constrait_len(tensor: torch.Tensor, max_len: int = 512) -> torch.Tensor:
    l, d = tensor.shape
    return v2.functional.resize(tensor.unsqueeze(0), (max_len, d)).squeeze(0) \
        if l > max_len else tensor
