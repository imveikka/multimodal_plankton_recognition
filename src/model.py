import torch
from torch import nn, Tensor
from typing import Dict, Any, Iterable
from image_encoder import ImageEncoder
from profile_encoder import ProfileTransformer
from lightning import LightningModule
from sklearn.preprocessing import LabelEncoder


class BiModal(LightningModule):


    def __init__(self, image_encoder_args: Dict[str, Any], 
                 profile_encoder_args: Dict[str, Any], 
                 dim_embed: int, class_names: Iterable[str]) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder(**image_encoder_args)
        self.profile_encoder = ProfileTransformer(**profile_encoder_args)

        self.image_projection = nn.Linear(self.image_encoder.dim_out, 
                                          dim_embed, bias=False)
        self.profile_projection = nn.Linear(self.profile_encoder.dim_out,
                                            dim_embed, bias=False)

        self.label_encoder = LabelEncoder().fit(class_names)


    def name_to_id(self, label: str | Iterable[str]) -> Tensor:
        if isinstance(label, str):
            label = [label]
        label = self.label_encoder.transform(label)
        return torch.tensor(label).long()
    

    def id_to_name(self, label: Tensor) -> Iterable:
        label = label.numpy()
        label = self.label_encoder.inverse_transform(label)
        return label


    def forward(self, image: Tensor, profile: Tensor, time: Tensor,
                padding_mask: Tensor, **kwargs) -> Dict[str, Tensor]:

        image_emb = self.image_encoder(image)
        profile_emb = self.profile_encoder(profile, time, padding_mask)

        image_emb = self.image_projection(image_emb)
        profile_emb = self.profile_projection(profile_emb)

        return {'image_emb': image_emb, 'profile_emb': profile_emb}
