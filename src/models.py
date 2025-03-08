import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.v2 as v2
from torchmetrics.functional import pairwise_cosine_similarity
from typing import Any
import math


class TS_BERT(nn.Module):

    def __init__(self, dim_in: int = 7, dim_emb: int = 512, num_head: int = 4, 
                 num_layers: int = 4, dim_feedforward: int = 1024, 
                 dropout: float = 0.1, max_len: int = 1200) -> None:
        super().__init__()

        self.embedding = nn.Linear(dim_in, dim_emb, bias=True)

        idx = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_emb, 2) * (-math.log(max_len) / dim_emb))
        position = torch.zeros(max_len, dim_emb)
        position[:, 0::2] = torch.sin(idx * div_term)
        position[:, 1::2] = torch.cos(idx * div_term)
        position = torch.cat((torch.zeros(1, dim_emb), position), 0)

        self.register_buffer('position', position)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_emb, nhead=num_head, 
                dim_feedforward=dim_feedforward,
                dropout=dropout, activation='gelu', batch_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = self.embedding(signal) + self.position[: signal.shape[1]]
        mask = (signal == 0).all(2)
        mask[:, 0] = True
        x = self.encoder(x, src_key_padding_mask=mask)
        return x[:, 0]


class CLIP(nn.Module):

    def __init__(self, model_1: nn.Module, model_2: nn.Module) -> None:
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.logit_scale = nn.Parameter(torch.ones([]))

    def loss(self, encoding_1: torch.Tensor, encoding_2: torch.Tensor) -> torch.Tensor:
        logits = 1 - pairwise_cosine_similarity(encoding_1, encoding_2) * self.logit_scale
        labels = torch.arange(logits.shape[0]).to(logits.device)
        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_s = nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_s) / 2
        return loss
    
    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor) -> tuple[torch.Tensor]:
        encoding_1 = self.model_1(input_1)
        encoding_2 = self.model_2(input_2)
        loss = self.loss(encoding_1, encoding_2)
        return encoding_1, encoding_2, loss



