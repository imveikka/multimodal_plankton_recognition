from pandas.core import base
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, functional as F
from typing import Iterable, Dict


class ProfileTransformer(Module):


    def __init__(self, dim_in: int, dim_hidden: int, target_size: int,
                 num_head: int, num_layers: int = 6,
                 dim_feedforward: int = 2024, dropout: float = 0.1, 
                 activation: str = 'gelu', metadata: bool = True) -> None:
        super().__init__()

        self.expand = nn.Linear(dim_in, dim_hidden, bias=False)
        self.position = nn.Embedding(target_size + 2, dim_hidden, padding_idx=-1)
        self.padding_idx = self.position.padding_idx

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_hidden, nhead=num_head, 
                dim_feedforward=dim_feedforward,
                dropout=dropout, activation=activation, 
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.drop = nn.Dropout(dropout)

        self.dim_out = dim_hidden + metadata
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
                padding_mask: Tensor, **kwargs) -> Tensor:

        x = self.expand(profile)
        x = x + self.position(time)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = x[:, 0]
        if self.metadata:
            metadata = kwargs['profile_len'].to(profile.dtype)
            metadata /= profile.shape[1]
            x = torch.cat((x, metadata), 1)
        return self.drop(x)


class ProfileLSTM(Module):


    def __init__(self, dim_in: int, dim_hidden: int, num_layers: int,
                 dropout: float = 0.1, metadata: bool = True) -> None:
        super().__init__()

        self.expand = nn.Linear(dim_in, dim_hidden, bias=False)
        self.lstm = nn.LSTM(dim_hidden, dim_hidden, num_layers,
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
                **kwargs) -> Tensor:

        x = self.expand(profile)
        x, _ = self.lstm(x)
        x = x[torch.arange(x.shape[0]), last_idx]
        if self.metadata:
            metadata = kwargs['profile_len'].to(profile.dtype)
            metadata /= profile.shape[1]
            x = torch.cat((x, metadata), 1)

        return self.drop(x)


class _BasicBlock(Module):


    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 downsample: Module | None = None, groups: int = 1,
                 base_channels: int = 64) -> None:
        super().__init__()

        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class ProfileCNN(Module):

    """
    ResNet-based CNN, adapted from https://github.com/Lornatang/ResNet-PyTorch/blob/main/model.py
    """

    def __init__(self, dim_in, blocks: list[int], groups: int = 1,
                 block_type: type[_BasicBlock] = _BasicBlock,
                 base_channels: int = 32, dropout = 0.1,
                 metadata: bool = True) -> None:

        super().__init__()
        self.in_channels = self.base_channels = base_channels
        self.dilation = 1
        self.groups = groups

        self.conv1 = nn.Conv1d(dim_in, self.in_channels, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.layer1 = self._make_layer(blocks[0], base_channels, block_type, 1)
        self.layer2 = self._make_layer(blocks[1], base_channels * 2, block_type, 2)
        self.layer3 = self._make_layer(blocks[2], base_channels * 4, block_type, 2)
        self.layer4 = self._make_layer(blocks[3], base_channels * 8, block_type, 2)

        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout)

        self.dim_out = base_channels * 8 + metadata
        self.metadata = metadata


    def _make_layer(self, repeat_times: int,channels: int, 
                    block_type: type[_BasicBlock] = _BasicBlock,
                    stride: int = 1) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != channels * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, channels * block_type.expansion, 1, stride, 0, bias=False),
                nn.BatchNorm1d(channels * block_type.expansion),
            )

        layers = [block_type(self.in_channels, channels, stride, downsample,
                             self.groups, self.base_channels)]

        self.in_channels = channels * block_type.expansion
        for _ in range(1, repeat_times):
            layers.append(block_type(self.in_channels, channels, 1,
                                     None, self.groups, self.base_channels))

        return nn.Sequential(*layers)


    def tokenize(self, profile: Tensor | Iterable[Tensor]) -> Dict[str, Tensor]:
        if not isinstance(profile, (list, tuple)):
            profile = [profile]
        profile = torch.stack(profile)
        return {'profile': profile}


    def forward_features(self, profile: Tensor) -> Tensor:

        profile = profile.transpose(1, 2)
        out = self.conv1(profile)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


    def forward(self, profile: Tensor, **kwargs) -> Tensor:

        out = self.forward_features(profile)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        if self.metadata:
            metadata = kwargs['profile_len'].to(profile.dtype)
            metadata /= profile.shape[1]
            out = torch.cat((out, metadata), 1)

        return self.drop(out)

