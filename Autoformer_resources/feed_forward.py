import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float, dim_ff: int):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.dim_ff = dim_ff
        self.linear_1 = torch.nn.Linear(self.d_model, self.dim_ff)
        self.linear_2 = torch.nn.Linear(self.dim_ff, self.d_model)



    def forward(self, x):
        x = self.linear_2(torch.relu(self.linear_1(x)))
        return x
