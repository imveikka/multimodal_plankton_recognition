import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as v2
from torchvision import models
from torchmetrics.functional import pairwise_cosine_similarity
from typing import Any
import math


class TS_BERT(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, 
                 num_head: int = 6, num_layers: int = 6, 
                 dim_feedforward: int = 1024, dropout: float = 0.1, 
                 activation='gelu', max_len: int = 512) -> None:
        super().__init__()

        self.embedding = nn.Linear(dim_in, dim_out, bias=False)
        self.position = nn.Embedding(max_len + 2, dim_out, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(dim_out)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_out, nhead=num_head, 
                dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation, 
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.dim_out = dim_out

    def preprocess_signal(self, signal: torch.Tensor) -> tuple[torch.Tensor]:
        b, l, d = signal.shape
        signal = torch.cat((torch.zeros(b, 1, d), signal), 1)
        mask = (signal == 0).all(2)
        mask[:, 0] = False
        time = torch.tile(torch.arange(1, l+2), (b, 1))
        time[mask] = 0
        return signal, time, mask

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal, time, mask = self.preprocess_signal(signal)
        x = self.embedding_norm(self.embedding(signal) + self.position(time))
        x = self.encoder(x, src_key_padding_mask=mask)
        return x[:, 0]


class ImageEncoder(nn.Module):

    def __init__(self, name: str, weights: str = 'DEFAULT') -> None:
        super().__init__()

        self.backbone = getattr(models, name)(weights=weights)

        self.norm = nn.LayerNorm(self.dim_out)

        # ResNet
        if hasattr(self.backbone, 'fc'):
            self.dim_out = self.backbone.fc.in_features
            self.norm = nn.LayerNorm(self.dim_out)
            self.backbone.fc = Empty()

        # AlexNet, DenseNet, SqueezeNet, EfficientNet
        elif hasattr(self.backbone, 'classifier'):
            if name.startswith('squeezenet'):
                self.dim_out = 512 
                self.backbone.classifier = nn.AdaptiveAvgPool2d((1, 1))
            elif name.startswith('efficientnet'):
                self.dim_out = self.backbone.classifier[1].in_features
                self.backbone.classifier = Empty()
            else:
                self.dim_out = self.backbone.classifier.in_features
                self.backbone.classifier = Empty()

        # ViT
        elif hasattr(self.backbone, 'heads'):
            self.dim_out = self.backbone.hidden_dim
            self.backbone.heads = Empty()
            self.norm = Empty() # ViT already has layernorm in the end

        
    def forward(self, image):
        x = self.backbone(image)
        x = self.norm(x)
        return x


class CLIP(nn.Module):

    def __init__(self, model_1: nn.Module, model_2: nn.Module, dim_embed: int) -> None:
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.projection_1 = nn.Linear(self.model_1.dim_out, dim_embed, bias=False)
        self.projection_2 = nn.Linear(self.model_2.dim_out, dim_embed, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]))

    def encode_1(self, input):
        return self.projection_1(self.model_1(input))

    def encode_2(self, input):
        return self.projection_2(self.model_2(input))

    def loss(self, encoding_1: torch.Tensor, encoding_2: torch.Tensor) -> torch.Tensor:
        logits = (1 - pairwise_cosine_similarity(encoding_1, encoding_2)) * self.logit_scale
        labels = torch.arange(logits.shape[0]).long().to(logits.device)
        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_s = nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_s) / 2
        return loss
    
    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor) -> tuple[torch.Tensor]:
        encoding_1 = self.encode_1(input_1)
        encoding_2 = self.encode_2(input_2)
        loss = self.loss(encoding_1, encoding_2)
        return encoding_1, encoding_2, loss

class Empty(nn.Module):

    def forward(self, x):
        return x

