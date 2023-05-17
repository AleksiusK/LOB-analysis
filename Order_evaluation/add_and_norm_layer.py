import torch
from torch import nn

class Norm(nn.Module):
    def __init__(self, d_model: int, eps: float):
        super(Norm, self).__init__()
        self.d_model = d_model
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
