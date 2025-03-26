import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, functional as F
from typing import Iterable, Dict


class ProfileTransformer(Module):


    def __init__(self, dim_in: int, dim_out: int, 
                 num_head: int, num_layers: int = 6, 
                 dim_feedforward: int = 2024, dropout: float = 0.1, 
                 activation: str = 'gelu', max_len: int = 256) -> None:
        super().__init__()

        self.expand = nn.Conv1d(dim_in, dim_out, 3, 1, 1, bias=False)
        self.position = nn.Embedding(max_len + 2, dim_out, padding_idx=-1)
        self.padding_idx = self.position.padding_idx
        self.norm = nn.LayerNorm(dim_out)

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
    
    
    def tokenize(self, profile: Tensor | Iterable[Tensor]) -> dict:

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

        x = self.expand(profile.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.norm(x + self.position(time))
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        return x[:, 0]

