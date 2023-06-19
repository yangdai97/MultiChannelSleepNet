import math

import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model=128, dropout=0.2, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # pe:[1, 30, 128]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head, dim_feedforward=config.forward_hidden, dropout=config.dropout)
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model * 3)

        self.position_multi = PositionalEncoding(d_model=config.dim_model * 3, dropout=0.1)
        encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model * 3, nhead=config.num_head,dim_feedforward=config.forward_hidden, dropout=config.dropout)
        self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)

        self.fc1 = nn.Sequential(
            nn.Linear(config.pad_size * config.dim_model * 3, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.fc_hidden, config.num_classes)
        )

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x1 = self.position_single(x1)
        x2 = self.position_single(x2)
        x3 = self.position_single(x3)

        x1 = self.transformer_encoder_1(x1)     # (batch_size, 29, 128)
        x2 = self.transformer_encoder_2(x2)
        x3 = self.transformer_encoder_3(x3)

        x = torch.cat([x1, x2, x3], dim=2)

        x = self.drop(x)
        x = self.layer_norm(x)
        residual = x

        x = self.position_multi(x)
        x = self.transformer_encoder_multi(x)

        x = self.layer_norm(x + residual)       # residual connection

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x