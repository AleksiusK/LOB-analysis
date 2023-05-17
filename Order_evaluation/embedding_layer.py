import torch
from torch import nn


class Embed(nn.Module):
    def __init__(self, order_dim: int, d_model: int):
        super(Embed, self).__init__()
        ## Quote from Pytorch: "A simple lookup table that stores embeddings of a fixed dictionary and size."
        ## Input is a list of indices that correspond to embeddings.
        self.linear = nn.Linear(order_dim, d_model)
        self.floor = torch.floor

    def forward(self, x):
        x = self.linear(x)
        x = self.floor(x)
        return x
