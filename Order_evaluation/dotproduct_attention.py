import math

import torch
from torch import nn


class DotProductAttention(nn.Module):
    def __init__(self, d_model: int, head_dim: int, dropout: float):
        super(DotProductAttention, self).__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        weights = self.softmax(scores)

        if self.dropout is not None:
            weights = self.dropout(weights)

        return torch.bmm(weights, v)
