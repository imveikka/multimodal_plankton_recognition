import torch
from torch import nn, Tensor, optim
from typing import Dict, Any, Iterable, Callable
from image_encoder import ImageEncoder
from profile_encoder import ProfileTransformer, ProfileLSTM
from coordination import *
from lightning import LightningModule
from sklearn.preprocessing import LabelEncoder
from itertools import pairwise


class MultiModel(LightningModule):


    def __init__(self, dim_embed, image_encoder_args: Dict[str, Any], 
                 profile_encoder_args: Dict[str, Any],
                 classifier_args: Dict[str, Any],
                 coordination_args: Dict[str, Any],
                 optim_args: Dict[str, Any],
                 class_names: Iterable[str],
                 supervised_coordination: bool = False) -> None:
        super().__init__()

        # Construct encoders
        self.image_encoder = ImageEncoder(**image_encoder_args)
        self.image_projection = nn.Linear(self.image_encoder.dim_out,
                                          dim_embed, bias=False)
        
        if 'num_head' in profile_encoder_args:
            self.profile_encoder = ProfileTransformer(**profile_encoder_args)
        else:
            self.profile_encoder = ProfileLSTM(**profile_encoder_args)
        self.profile_projection = nn.Linear(self.profile_encoder.dim_out,
                                            dim_embed, bias=False)

        # Construct classifier
        hidden = tuple(classifier_args['dim_hidden_layers'])
        layers = (dim_embed,) + hidden + (len(class_names),)
        self.classifier_layers = nn.ModuleList([
            nn.Linear(dim_in, dim_out) for dim_in, dim_out in pairwise(layers)
        ])
        self.classifier_drop = nn.Dropout(p=classifier_args['dropout'])
        self.classifier_act = getattr(F, classifier_args['activation'])

        # Losses
        method = coordination_args.get('method')
        if method == 'clip':
            self.coor_loss = CLIPLoss()
        elif method == 'rank':
            self.coor_loss = RankLoss(margin=coordination_args['margin'])
        elif method == 'distance':
            self.coor_loss = DistanceLoss()
        else:
            self.coor_loss = Zero()
        self.clas_loss = nn.CrossEntropyLoss()

        # Miscellaneous
        self.label_encoder = LabelEncoder().fit(class_names)
        self.alpha = coordination_args['alpha']
        self.supervised_coordination = coordination_args['supervised']
        self.optim_args = optim_args


    def name_to_id(self, label: str | Iterable[str]) -> Tensor:
        if isinstance(label, str):
            label = [label]
        label = self.label_encoder.transform(label)
        return torch.tensor(label).long()
    

    def id_to_name(self, label: Tensor) -> Iterable:
        label = label.numpy()
        label = self.label_encoder.inverse_transform(label)
        return label

    
    def safe_forward(self, model: Callable, **kwargs):
        return model(**kwargs) if None not in kwargs.values() else None 
 

    def tokenize(self, profile: Tensor) -> Dict[str, Tensor]:
        return self.profile_encoder.tokenize(profile)    


    def encode(self, image: Tensor | None, profile: Tensor | None, 
               **kwargs) -> Dict[str, Tensor]:

        image_emb = self.safe_forward(self.image_encoder, image=image)
        profile_emb = self.safe_forward(self.profile_encoder, profile=profile,
                                        **kwargs)

        image_emb = self.safe_forward(self.image_projection,
                                      input=image_emb)
        profile_emb = self.safe_forward(self.profile_projection,
                                        input=profile_emb)

        return {'image_emb': image_emb, 'profile_emb': profile_emb}
    

    def classify_(self, x: Tensor | None):

        for layer in self.classifier_layers[:-1]:
            x = self.classifier_drop(x)
            x = layer(x)
            x = self.classifier_act(x)
    
        return self.classifier_layers[-1](x)


    def classify(self, image_emb: Tensor | None, profile_emb: Tensor | None,
                 **kwargs) -> Dict[str, Tensor]:
        
        logits_1 = self.safe_forward(self.classify_, x=image_emb)
        logits_2 = self.safe_forward(self.classify_, x=profile_emb)

        if logits_1 is None: logits_1 = 0
        if logits_2 is None: logits_2 = 0

        return logits_1 + logits_2
    
    
    def forward(self, **kwargs) -> Dict[str, Tensor]:
        
        embeddings = self.encode(**kwargs)
        logits = self.classify(**embeddings)

        return logits


    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        embeddings = self.encode(**batch)
        embeddings_stacked = torch.cat(list(embeddings.values()))
        label_stacked = torch.tile(batch['label'], (2,))
        logits = self.classify_(embeddings_stacked)

        loss_coor = self.coor_loss(
            **embeddings, 
            label=batch['label'] if self.supervised_coordination else None
        )
        loss_clas = self.clas_loss(logits, label_stacked)
        loss = self.alpha * loss_coor + (1 - self.alpha) * loss_clas

        metrics = {
            'Train/loss_total': loss,
            'Train/loss_coord': loss_coor,
            'Train/loss_class': loss_clas,
        }
        self.log_dict(metrics)

        return loss

    
    def validation_step(self, batch: Dict[str, Tensor],
                        batch_idx: int) -> Tensor:

        embeddings = self.encode(**batch)
        embeddings_stacked = torch.cat(list(embeddings.values()))
        label_stacked = torch.tile(batch['label'], (2,))
        logits = self.classify_(embeddings_stacked)

        loss_coor = self.coor_loss(
            **embeddings, 
            label=batch['label'] if self.supervised_coordination else None
        )
        loss_clas = self.clas_loss(logits, label_stacked)
        loss = self.alpha * loss_coor + (1 - self.alpha) * loss_clas

        metrics = {
            'Valid/loss_total': loss,
            'Valid/loss_coord': loss_coor,
            'Valid/loss_class': loss_clas,
        }
        self.log_dict(metrics, on_epoch=True)

       
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), **self.optim_args)


class ImageModel(LightningModule):


    def __init__(self, image_encoder_args: Dict[str, Any], 
                 optim_args: Dict[str, Any],
                 class_names: Iterable[str]) -> None:
        super().__init__()

        # Construct encoders
        self.image_model = ImageEncoder(**image_encoder_args)

        # loss
        self.clas_loss = nn.CrossEntropyLoss()

        # Miscellaneous
        self.label_encoder = LabelEncoder().fit(class_names)
        self.optim_args = optim_args


    def name_to_id(self, label: str | Iterable[str]) -> Tensor:
        if isinstance(label, str):
            label = [label]
        label = self.label_encoder.transform(label)
        return torch.tensor(label).long()
    

    def id_to_name(self, label: Tensor) -> Iterable:
        label = label.numpy()
        label = self.label_encoder.inverse_transform(label)
        return label


    def forward(self, image, **kwargs) -> Dict[str, Tensor]:
        logits = self.image_model(image)
        return {'logits': logits}


    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        logits = self(**batch)['logits']
        label = batch['label']

        loss = self.clas_loss(logits, label)

        metrics = {'Train/loss': loss}
        self.log_dict(metrics)

        return loss

    
    def validation_step(self, batch: Dict[str, Tensor],
                        batch_idx: int) -> Tensor:

        logits = self(**batch)['logits']
        label = batch['label']

        loss = self.clas_loss(logits, label)

        metrics = {'Valid/loss': loss}
        self.log_dict(metrics, on_epoch=True)

       
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), **self.optim_args)


class ProfileModel(LightningModule):


    def __init__(self, profile_encoder_args: Dict[str, Any], 
                 optim_args: Dict[str, Any],
                 class_names: Iterable[str]) -> None:
        super().__init__()

        # Construct encoders
        if 'num_head' in profile_encoder_args:
            self.profile_encoder = ProfileTransformer(**profile_encoder_args)
        else:
            self.profile_encoder = ProfileLSTM(**profile_encoder_args)

        # classifier
        self.fc = nn.Linear(self.profile_encoder.dim_out, len(class_names))

        # loss
        self.clas_loss = nn.CrossEntropyLoss()

        # Miscellaneous
        self.label_encoder = LabelEncoder().fit(class_names)
        self.optim_args = optim_args


    def name_to_id(self, label: str | Iterable[str]) -> Tensor:
        if isinstance(label, str):
            label = [label]
        label = self.label_encoder.transform(label)
        return torch.tensor(label).long()
    

    def id_to_name(self, label: Tensor) -> Iterable:
        label = label.numpy()
        label = self.label_encoder.inverse_transform(label)
        return label


    def tokenize(self, profile: Tensor) -> Dict[str, Tensor]:
        return self.profile_encoder.tokenize(profile)    


    def forward(self, profile, **kwargs) -> Dict[str, Tensor]:
        embeddings = self.profile_encoder(profile, **kwargs)
        logits = self.fc(embeddings)
        return {'logits': logits}


    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        logits = self(**batch)['logits']
        label = batch['label']

        loss = self.clas_loss(logits, label)

        metrics = {'Train/loss': loss}
        self.log_dict(metrics)

        return loss

    
    def validation_step(self, batch: Dict[str, Tensor],
                        batch_idx: int) -> Tensor:

        logits = self(**batch)['logits']
        label = batch['label']

        loss = self.clas_loss(logits, label)

        metrics = {'Valid/loss': loss}
        self.log_dict(metrics, on_epoch=True)

       
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), **self.optim_args)

