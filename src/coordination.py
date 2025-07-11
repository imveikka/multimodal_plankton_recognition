import torch
from torch import Tensor, clip, nn
from torch.nn import MSELoss, Module, Parameter, functional as F
import math


class DistanceLoss(Module):
    

    def forward(self, image_emb: Tensor, profile_emb: Tensor,
                label: Tensor | None = None) -> Tensor:
        residuals = (image_emb - profile_emb).pow(2)
        loss = residuals.mean()
        return loss


class CLIPLoss(Module):

    """ From https://arxiv.org/pdf/2103.00020 """

    def __init__(self, bias: bool = False) -> None:
        super().__init__()
        self.logit_scale = Parameter(torch.ones([]))
    

    def forward(self, image_emb: Tensor, profile_emb: Tensor,
                buckets: int = 1) -> Tensor:

        assert image_emb.size(0) % buckets == 0, \
        "Batch size must be divisible by number of buckets!"
        bucket_size = image_emb.size(0) // buckets

        image_emb = F.normalize(image_emb)
        profile_emb = F.normalize(profile_emb)

        image_emb = image_emb.view(buckets, bucket_size, -1)
        profile_emb = profile_emb.view(buckets, bucket_size, -1)
        logits = (image_emb @ profile_emb.transpose(1, 2)) * self.logit_scale.exp()

        label = torch.arange(bucket_size).long().to(image_emb.device)
        label = torch.stack([label] * buckets)

        loss_1 = torch.stack([F.cross_entropy(x, y) for x, y in zip(logits, label)]).mean()
        loss_2 = torch.stack([F.cross_entropy(x.T, y) for x, y in zip(logits, label)]).mean()
        loss = (loss_1 + loss_2) / 2

        return loss


class CLIPPlus(Module):


    def __init__(self) -> None:
        super().__init__()
        self.clip = CLIPLoss()
        self.l2 = MSELoss()
    

    def forward(self, image_emb: Tensor, profile_emb: Tensor) -> Tensor:
        loss_1 = self.clip(image_emb, profile_emb)
        loss_2 = self.l2(image_emb, profile_emb)
        return loss_1 + loss_2


class SigLIPLoss(Module):


    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = Parameter(torch.ones([]))
        self.bias = Parameter(-10 * torch.ones([]))


    def forward(self, image_emb: Tensor, profile_emb: Tensor,
                buckets: int = 1) -> Tensor:

        assert image_emb.size(0) % buckets == 0, \
        "Batch size must be divisible by number of buckets!"

        image_emb = F.normalize(image_emb)
        profile_emb = F.normalize(profile_emb)

        bucket_size = image_emb.size(0) // buckets
        image_emb = image_emb.view(buckets, bucket_size, -1)
        profile_emb = profile_emb.view(buckets, bucket_size, -1)

        logits = (image_emb @ profile_emb.transpose(1, 2)) * self.logit_scale.exp() + self.bias
        logits = logits * (-1)
        logits.diagonal(0, 1, 2).mul_(-1)

        loss = -F.logsigmoid(logits).sum((1, 2)) / bucket_size

        return loss.mean()


class RankLoss(Module):


    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin


    def forward(self, image_emb: Tensor, profile_emb: Tensor) -> Tensor:

        image_emb = image_emb / image_emb.norm(dim=1, keepdim=True)
        profile_emb = profile_emb / profile_emb.norm(dim=1, keepdim=True)       

        logits = (image_emb @ profile_emb.T)
        logits.diagonal().mul_(-1)

        loss_1 = F.relu(self.margin + logits.sum(0)).mean()
        loss_2 = F.relu(self.margin + logits.sum(1)).mean()
        loss = (loss_1 + loss_2) / 2

        return loss
    

class Zero(Module):

    
    def forward(self, **kwargs):
        return 0


class ArcFace(Module):

    r"""
        ArcFace, from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    
        Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0,
                 m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m


    def forward(self, image_emb, profile_emb, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        emb = torch.concat((image_emb, profile_emb))
        label = torch.tile(label, (2,))
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return F.cross_entropy(output, label)

