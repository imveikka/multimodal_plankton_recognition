import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as v2
from torchvision import models
from typing import Any
import math


class Empty(nn.Module):

    def forward(self, x):
        return x


class TS_Transformer(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, 
                 num_head: int = 6, num_layers: int = 6, 
                 dim_feedforward: int = 1024, dropout: float = 0.1, 
                 activation='gelu', max_len: int = 256, 
                 is_causal: bool = False) -> None:
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
        self.is_causal = is_causal

    def preprocess_signal(self, signal: torch.Tensor) -> tuple[torch.Tensor]:
        b, l, d = signal.shape
        CLS = torch.zeros(b, 1, d).to(signal.device)
        signal = torch.cat((CLS, signal), 1)
        mask = (signal == 0).all(2)
        mask[:, 0] = False
        time = torch.tile(torch.arange(1, l+2), (b, 1)).to(signal.device)
        time[mask] = 0
        return signal, time, mask

    """ From CLIP GitHub """
    def build_attention_mask(self, batch_size: int):
        # lazily create causal attention mask, with full attention between 
        # the vision tokens pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(batch_size, batch_size)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal, time, padding_mask = self.preprocess_signal(signal)
        x = self.embedding_norm(self.embedding(signal) + self.position(time))
        causal_mask = self.build_attention_mask(signal.shape[0]) \
            if self.is_causal else None
        x = self.encoder(x, mask=causal_mask, 
                         src_key_padding_mask=padding_mask,
                         is_causal=self.is_causal)
        return x[:, 0]


class ImageEncoder(nn.Module):

    def __init__(self, name: str, weights: str | None = None) -> None:
        super().__init__()

        self.backbone = getattr(models, name)(weights=weights)

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
            self.norm = nn.LayerNorm(self.dim_out)

        # ViT
        elif hasattr(self.backbone, 'heads'):
            self.dim_out = self.backbone.hidden_dim
            self.backbone.heads = Empty()
            self.norm = Empty() # ViT already has layernorm in the end

        
    def forward(self, image):
        x = self.backbone(image)
        x = self.norm(x)
        return x


class BiModal(nn.Module):

    def __init__(self, model_1: nn.Module, model_2: nn.Module, 
                 dim_embed: int) -> None:
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.projection_1 = nn.Linear(self.model_1.dim_out, dim_embed, bias=False)
        self.projection_2 = nn.Linear(self.model_2.dim_out, dim_embed, bias=False)

    def encode_1(self, input):
        return self.projection_1(self.model_1(input))

    def encode_2(self, input):
        return self.projection_2(self.model_2(input))

    def forward(self, input_1: torch.Tensor, 
                input_2: torch.Tensor) -> tuple[torch.Tensor]:
        encoding_1 = self.encode_1(input_1)
        encoding_2 = self.encode_2(input_2)
        return encoding_1, encoding_2


class CLIPLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
    
    def forward(self, encoding_1: torch.Tensor, encoding_2: torch.Tensor, 
                y: None|torch.Tensor = None) -> torch.Tensor:
        encoding_1 = encoding_1 / encoding_1.norm(dim=1, keepdim=True)
        encoding_2 = encoding_2 / encoding_2.norm(dim=1, keepdim=True)       
        logits = (encoding_1 @ encoding_2.T) * self.logit_scale.exp()
        if y is None:
            labels = torch.arange(encoding_1.shape[0]).long().to(encoding_1.device)
        else:
            pass
        loss_1 = nn.functional.cross_entropy(logits, labels)
        loss_2 = nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_1 + loss_2) / 2
        return loss


class RankLoss(nn.Module):

    def __init__(self, margin: float) -> None:
        self.margin = margin

    def forward(self, encoding_1: torch.Tensor, 
                encoding_2: torch.Tensor) -> torch.Tensor:
        encoding_1 = encoding_1 / encoding_1.norm(dim=1, keepdim=True)
        encoding_2 = encoding_2 / encoding_2.norm(dim=1, keepdim=True)       
        logits = (encoding_1 @ encoding_2.T)

        logits.diagonal().mul_(-1)
        loss_1 = nn.functional.relu(self.margin + logits.sum(0)).sum()
        loss_2 = nn.functional.relu(self.margin + logits.sum(1)).sum()
        loss = (loss_1 + loss_2) / 2

class DistanceLoss(nn.Module):

    def
        
