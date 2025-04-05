import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, functional as F
from typing import Iterable, Dict


class ProfileTransformer(Module):


    def __init__(self, dim_in: int, max_len: int,
                 num_head: int, num_layers: int = 6,
                 dim_feedforward: int = 2024, dropout: float = 0.1, 
                 activation: str = 'gelu', metadata: bool = True) -> None:
        super().__init__()

        self.dim_out = dim_in + (dim_in % num_head)
        self.pad = nn.ZeroPad2d((0, 1, 0, 0))
        self.position = nn.Embedding(max_len + 2, self.dim_out, padding_idx=-1)
        self.padding_idx = self.position.padding_idx

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.dim_out, nhead=num_head, 
                dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation, 
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.drop = nn.Dropout(dropout)

        self.dim_out += metadata
        self.metadata = metadata
    
    
    def tokenize(self, profile: Tensor | Iterable[Tensor]) -> Dict[str, Tensor]:

        if not isinstance(profile, (list, tuple)):
            profile = [profile]

        time = [torch.arange(0, 1 + p.shape[0]) for p in profile]
        time = nn.utils.rnn.pad_sequence(time, batch_first=True, 
                                         padding_value=self.padding_idx)

        profile = nn.utils.rnn.pad_sequence(profile, batch_first=True)
        profile = F.pad(profile, (0, 0, 1, 0)) # CLS token

        padding_mask = torch.empty(time.shape, dtype=torch.bool)
        padding_mask.fill_(False)
        padding_mask[time == self.padding_idx] = True

        return {'profile': profile, 'time': time, 'padding_mask': padding_mask}


    def forward(self, profile: Tensor, time: Tensor,
                padding_mask: Tensor, **kwargs) -> Dict[str, Tensor]:

        x = self.pad(profile)
        x = x + self.position(time)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x[:, 0]
        if self.metadata:
            metadata = kwargs['profile_len'].to(profile.dtype)
            metadata /= profile.shape[1]
            x = torch.cat((x, metadata), 1)
        return self.drop(x)



class ProfileLSTM(nn.Module):


    def __init__(self, dim_in: int, dim_hidden: int, num_layers: int,
                 dropout: float = 0.1, metadata: bool = True) -> None:
        super().__init__()

        self.lstm = nn.LSTM(dim_in, dim_hidden, num_layers,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.dim_out = dim_hidden + metadata
        self.metadata = metadata


    def tokenize(self, profile: Tensor | Iterable[Tensor]) -> Dict[str, Tensor]:

        if not isinstance(profile, (list, tuple)):
            profile = [profile]
        
        last = torch.tensor([p.shape[0] - 1 for p in profile]).long()
        profile = nn.utils.rnn.pad_sequence(profile, batch_first=True)

        return {'profile': profile, 'last_idx': last}


    def forward(self, profile: Tensor, last_idx: Tensor,
                **kwargs) -> Dict[str, Tensor]:

        x, _ = self.lstm(profile)
        x = x[torch.arange(x.shape[0]), last_idx]
        if self.metadata:
            metadata = kwargs['profile_len'].to(profile.dtype)
            metadata /= profile.shape[1]
            x = torch.cat((x, metadata), 1)

        return self.drop(x)