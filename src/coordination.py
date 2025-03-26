import torch
from torch import Tensor
from torch.nn import Module, Parameter, functional as F


class DistanceLoss(Module):
    

    def forward(self, image_emb: Tensor, profile_emb: Tensor) -> Tensor:
        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        profile_emb = profile_emb / profile_emb.norm(dim=1, keepdim=True)       
        residuals = torch.norm(image_emb - profile_emb, dim=1).pow(2)
        loss = residuals.mean()
        return loss


class CLIPLoss(Module):

    """ From https://arxiv.org/pdf/2103.00020 """

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = Parameter(torch.ones([]))
    

    def forward(self, image_emb: Tensor, profile_emb: Tensor,
                label: None | Tensor = None) -> Tensor:

        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        profile_emb = profile_emb / profile_emb.norm(dim=1, keepdim=True)       
        logits = (image_emb @ profile_emb.T) * self.logit_scale.exp()

        if label is not None:
            label = (label == label.reshape(-1, 1)).float()
        else:
            label = torch.arange(image_emb.shape[0]).long().to(image_emb.device)

        loss_1 = F.cross_entropy(logits, label)
        loss_2 = F.cross_entropy(logits.T, label)
        loss = (loss_1 + loss_2) / 2

        return loss


class RankLoss(Module):


    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin


    def forward(self, image_emb: Tensor, profile_emb: Tensor,
                label: None | Tensor = None) -> Tensor:

        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        profile_emb = profile_emb / profile_emb.norm(dim=1, keepdim=True)       
        logits = (image_emb @ profile_emb.T)

        if label is not None:
            logits[label == label.reshape(-1, 1)] *= -1
        else:
            logits.diagonal().mul_(-1)

        loss_1 = F.relu(self.margin + logits.sum(0)).mean()
        loss_2 = F.relu(self.margin + logits.sum(1)).mean()
        loss = (loss_1 + loss_2) / 2

        return loss