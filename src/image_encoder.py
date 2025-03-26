from torch import nn, Tensor
from torch.nn import Module
import timm 
from typing import Dict


class ImageEncoder(Module):


    def __init__(self, name: str, num_classes: int = 0,
                 pretrained: bool = False) -> None:
        super().__init__()

        self.backbone = timm.create_model(name, num_classes=num_classes, 
                                          pretrained=False)
        self.dim_out = self.backbone.num_features
        self.norm = nn.LayerNorm(self.dim_out)


    def forward(self, image: Tensor, **kwargs) -> Dict[str, Tensor]:
        x = self.backbone(image)
        x = self.norm(x)
        return x
