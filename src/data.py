import torch
from torch import nn, Tensor, functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2

import os
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Dict


class MultiSet(Dataset):


    def __init__(self, annotation_path: Path, image_transforms: object,
                 profile_transform: object, pair_augmentation: object) -> None:
        super().__init__()

        self.parent = annotation_path.parent
        self.table = pd.read_csv(annotation_path)

        self.X = self.table.X.to_numpy()
        self.image_files = [self.parent / f'images/{x}.jpg' for x in self.X]
        self.profile_files = [self.parent / f'profiles/{x}.csv' for x in self.X]
        self.labels = self.table.class_name.to_numpy()
        self.class_names = np.unique(self.labels)
        
        self.image_transforms = image_transforms
        self.profile_transform = profile_transform
        self.pair_augmentation = pair_augmentation


    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        image = torchvision.io.decode_image(self.image_files[index])
        profile = np.loadtxt(self.profile_files[index], delimiter=',', skiprows=1)  
        profile = torch.tensor(profile)

        image = self.image_transforms(image)
        profile = self.profile_transform(profile)
        label = self.labels[index]

        if self.pair_augmentation:
            image, profile = self.pair_augmentation(image, profile)
        else:
            image = v2.functional.resize(image, (224, 224))

        return {'image': image, 'profile': profile, 'label': label}


class ImageTransforms(object):

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        bg = find_background_color(image)
        image[0, :25, :] = bg[0]
        image[1, :25, :] = bg[1]
        image[2, :25, :] = bg[2]
        image = square_pad(image, fill=bg)
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        return image


class ProfileTransform(object):


    def __init__(self, max_len: int = None):
        self.min = torch.tensor([0, 0, 0, 0, 0, 0, -1])
        self.diff = torch.tensor([14850, 7360, 408, 7360, 7488, 7488, 2])
        self.max_len = max_len


    def __call__(self, profile: torch.Tensor) -> torch.Tensor:
        profile = (profile - self.min) / self.diff
        profile = profile.float()
        if self.max_len:
            profile = constrait_len(profile, max_len=self.max_len)
        return profile


class PairAugmentation(object):


    def __init__(self):
        self.expand = v2.Resize((240, 240))
        self.crop = v2.RandomCrop((224, 224))
        self.jitter = v2.ColorJitter(0.3, 0.3)


    def __call__(self, image: torch.Tensor, profile: torch.Tensor) -> tuple[torch.Tensor]:
        bg = find_background_color(image)
        image = self.expand(image)
        image = self.crop(image)
        image += (5e-3 * torch.randn(image.shape[1:]))
        profile += (5e-3 * torch.randn(profile.shape))
        image = self.jitter(image)
        if random.randint(0, 1) == 0:
            image = v2.functional.vertical_flip(image)
        if random.randint(0, 1) == 0:
            image = v2.functional.horizontal_flip(image)
            profile = profile.flip(0)
        return image, profile


def find_background_color(image: Tensor, p: int = 2) -> Iterable[int]:

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


def square_pad(image: Tensor, fill: int | Iterable[int] = 0) -> Tensor:

    """
    Pads an image to square shape.
    """

    c, h, w = image.shape
    d = h - w
    if d < 0: padding = (0, -d // 2 + (-d % 2), 0, -d // 2)
    elif d > 0: padding = (d // 2 + (d % 2), 0, d // 2, 0)
    else: padding = (0, 0)
    return v2.functional.pad(image, padding=padding, fill=fill)


def constrait_len(profile: Tensor, max_len: int = 512) -> Tensor:
    l, d = profile.shape
    return v2.functional.resize(profile.unsqueeze(0), (max_len, d)).squeeze(0) \
        if l > max_len else profile
