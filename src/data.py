import torch
from torch import nn, Tensor, functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2

import os
import math
import random
import numpy as np
import cv2
from scipy.stats import mode
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
        self.image_files = [self.parent / f'../images/{x}.jpg' for x in self.X]
        self.profile_files = [self.parent / f'../profiles/{x}.csv' for x in self.X]
        self.labels = self.table.class_name.to_numpy()
        self.class_names = np.unique(self.labels)
        
        self.image_transforms = image_transforms
        self.profile_transform = profile_transform
        self.pair_augmentation = pair_augmentation


    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)
        profile = np.loadtxt(self.profile_files[index], delimiter=',', skiprows=1)  
        profile = profile[:, [0, 1, 3, 4, 5]]
        profile = torch.tensor(profile)

        image_shape = torch.tensor(image.shape)
        profile_length = torch.tensor([profile.shape[0]])

        image = self.image_transforms(image)
        profile = self.profile_transform(profile)
        label = self.labels[index]


        if self.pair_augmentation:
            image, profile = self.pair_augmentation(image, profile)
        else:
            image = v2.functional.resize(image, (224, 224))

        return {'image': image, 'profile': profile, 'label': label,
                'image_shape': image_shape,
                'profile_length': profile_length}


class ImageTransforms(object):

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        bg, std = find_background_stats(image)
        image = cover_scale(image, bg, std)
        image = pad_image_to_square(image, bg, std)
        image = torch.tensor(np.stack((image,) * 3))
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        return image


class ProfileTransform(object):


    def __init__(self, max_len: int = None):
        # self.min = torch.tensor([0, 0, 0, 0, 0, 0, -1])
        # self.diff = torch.tensor([14850, 7360, 408, 7360, 7488, 7488, 2])
        self.max_len = max_len


    def __call__(self, profile: torch.Tensor) -> torch.Tensor:
        # profile = (profile - self.min) / self.diff
        profile = profile.add(1).log()
        profile = profile.float()
        if self.max_len:
            profile = resize_profile(profile, max_len=self.max_len)
        return profile


class PairAugmentation(object):


    def __init__(self):
        self.resize = v2.Resize((240, 240))
        self.affine = v2.RandomAffine(degrees=(-2, 2),
                                      translate=(.02, .02),
                                      scale=(.98, 1.02))
        self.crop = v2.RandomCrop((224, 224))
        self.jitter = v2.ColorJitter(0.1, 0.1)


    def __call__(self, image: torch.Tensor, profile: torch.Tensor) -> tuple[torch.Tensor]:

        image = self.resize(image)
        image = self.affine(image)
        image = self.crop(image)
        image += (1e-3 * torch.randn(image.shape[1:]))
        profile += (1e-3 * torch.randn(profile.shape))
        image = self.jitter(image)
        if random.randint(0, 1) == 0:
            image = v2.functional.vertical_flip(image)
        if random.randint(0, 1) == 0:
            image = v2.functional.horizontal_flip(image)
            profile = profile.flip(0)

        return image, profile


def cover_scale(image: np.ndarray, bg: np.ndarray,
                std: np.ndarray) -> np.ndarray:
    noise = np.random.normal(loc=bg, scale=std, size=image[:25].shape).astype(image.dtype)
    image[:25] = noise
    return image


def find_background_stats(image: Tensor, p: int = 2,
                          closest: float = 0.90) -> Iterable[int]:

    """
    Finds the background statistics from image. 
    Based on mode from image rim of thickness p (pixels).
    Can be then used to fill background with random gaussian noise.
    """

    c = 1 if len(image.shape) < 3 else image.shape[-1]

    edges = np.concat(
        [
            image[:, :p].reshape(-1, c),
            image[:, :-p].reshape(-1, c),
            image[:p, :].reshape(-1, c),
            image[-p:, :].reshape(-1, c),
        ], 
        axis=0
    )

    color_mode = mode(edges, axis=0).mode
    n_closest = int(edges.shape[0] * closest)
    distances = np.sum((edges - color_mode)**2, axis=1)
    closest_indices = np.argpartition(distances, n_closest)[:n_closest]
    color_std = np.std(edges[closest_indices].astype(float), axis=0)

    return color_mode, color_std


def pad_image_to_square(image: np.ndarray, bg: np.ndarray,
                        std: np.ndarray) -> np.ndarray:

    height, width = image.shape[:2]
    
    # Calculate new size and padding
    max_side = max(height, width)
    y_from = (max_side - height) // 2
    x_from = (max_side - width) // 2

    if x_from > 0 or y_from > 0:
        # Create a new image with padding
        new_image = np.full((max_side, max_side), fill_value=bg, dtype=image.dtype)
        noise = np.random.normal(loc=0, scale=std, size=new_image.shape).astype(image.dtype)
        output_img = np.clip(new_image + noise, 0, 255).astype(image.dtype)
        # Place the original image in the center
        output_img[y_from:y_from+height, x_from:x_from+width] = image
    else:
        output_img = image
        
    return output_img


def constrait_len(profile: Tensor, max_len: int = 512) -> Tensor:
    l, d = profile.shape
    return v2.functional.resize(profile.unsqueeze(0), (max_len, d)).squeeze(0) \
        if l > max_len else profile


def resize_profile(profile: Tensor, max_len: int = 512) -> Tensor:
    _, d = profile.shape
    return v2.functional.resize(profile.unsqueeze(0), (max_len, d)).squeeze(0)
