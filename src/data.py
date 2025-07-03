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
from PIL import Image


class MultiSet(Dataset):


    def __init__(self, annotation_path: Path, image_transforms: object,
                 profile_transform: object, pair_augmentation: object) -> None:
        super().__init__()

        self.parent = annotation_path.parent
        self.table = pd.read_csv(annotation_path)

        self.id = self.table.ID.to_numpy()
        self.image_files = [self.parent / f'../images/{id}.jpg' for id in self.id]
        self.profile_files = [self.parent / f'../profiles/{id}.csv' for id in self.id]
        self.labels = self.table.class_name.to_numpy()
        self.class_names = np.unique(self.labels)
        
        self.image_transforms = image_transforms
        self.profile_transform = profile_transform
        self.pair_augmentation = pair_augmentation


    def __len__(self):
        return len(self.id)
    

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        image = cv2.imread(self.image_files[index], cv2.IMREAD_GRAYSCALE)
        profile = np.loadtxt(self.profile_files[index], delimiter=',', skiprows=1)  
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


class ImageTransforms:

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        bg, std = find_background_stats(image)
        image = cover_scale(image, bg, std)
        image = pad_image_to_square(image, bg, std)
        image = torch.tensor(image).unsqueeze(0)
        image = v2.functional.to_dtype(image, dtype=torch.float32, scale=True)
        return image


class ProfileTransform:


    def __init__(self, max_len: int = None):
        self.max_len = max_len


    def __call__(self, profile: torch.Tensor) -> torch.Tensor:
        profile = profile.add(1).log().float()
        if self.max_len:
            profile = resize_profile(profile, max_len=self.max_len)
        return profile


class FixedHeightResize:


    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        w, h = img.size
        aspect_ratio = float(h) / float(w)
        new_w = math.ceil(self.size / aspect_ratio)
        return F.resize(img, (self.size, new_w))


class PairAugmentation:


    def __init__(self):
        self.resize = v2.Resize((224, 224))
        self.affine = v2.RandomAffine(degrees=(-2, 2),
                                      translate=(.02, .02),
                                      scale=(.98, 1.02))
        self.jitter = v2.ColorJitter(0.1, 0.1)


    def __call__(self, image: torch.Tensor, profile: torch.Tensor) -> tuple[torch.Tensor]:

        image = self.resize(image)
        image += (1e-3 * torch.randn(image.shape[1:]))
        profile += (1e-3 * torch.randn(profile.shape))
        image = self.jitter(image)
        if random.randint(0, 1) == 0:
            image = v2.functional.vertical_flip(image)
        if random.randint(0, 1) == 0:
            image = v2.functional.horizontal_flip(image)
            profile = profile.flip(0)
        image = self.affine(image)

        return image, profile


def cover_scale(image: np.ndarray, bg: np.ndarray,
                std: np.ndarray) -> np.ndarray:
    noise = np.random.normal(loc=bg, scale=std, size=image[:25].shape).astype(image.dtype)
    image[:25] = noise
    return image


def find_background_stats(image: Tensor, p: int = 2,
                          closest: float = 0.80) -> Iterable[int]:

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


def resize_pil(img: Image, target_res: int = 224,
           resize: bool = True, edge: bool = False) -> Image:

    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    return Image.fromarray(canvas)


def constrait_len(profile: Tensor, max_len: int = 512) -> Tensor:
    l, d = profile.shape
    return v2.functional.resize(profile.unsqueeze(0), (max_len, d)).squeeze(0) \
        if l > max_len else profile


def resize_profile(profile: Tensor, target_len: int = 256) -> Tensor:
    _, d = profile.shape
    profile = profile.unsqueeze(0)
    profile = v2.functional.resize(profile, (target_len, d))
    return profile.squeeze(0)

