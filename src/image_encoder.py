import torch
from torch import nn, Tensor
from torch.nn import Module
import timm 
from typing import Dict


class ImageEncoder(Module):


    def __init__(self, name: str, num_classes: int = 0,
                 pretrained: bool = False, metadata: bool = True) -> None:
        super().__init__()

        self.backbone = timm.create_model(name, num_classes=num_classes, 
                                          pretrained=False)
        self.dim_out = self.backbone.num_features + metadata
        self.norm = nn.LayerNorm(self.dim_out)
        self.metadata = metadata


    def forward(self, image: Tensor, **kwargs) -> Dict[str, Tensor]:
        x = self.backbone(image)
        x = self.norm(x)
        if self.metadata:
            metadata = kwargs['image_shape'].to(image.dtype)
            metadata /= image.shape[2]
            x = torch.cat((x, metadata), 1)
        return x
