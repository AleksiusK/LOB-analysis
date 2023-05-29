import torch
import torch.nn as nn


class Embed(nn.Module):
    def __init__(self, horizon: int, d_model: int, input_length: int):
        super(Embed, self).__init__()
        self.d_model = d_model
        self.input_length = input_length

    def build(self, input_shape):
        self.W = nn.Parameter(torch.randn(input_shape[-1], self.d_model - input_shape[2]).uniform_())
        self.P = nn.Parameter(torch.randn(1, input_shape[1], self.d_model - input_shape[2]).uniform_())
        self.w = nn.Parameter(torch.randn(input_shape[1], 1).uniform_())
        self.p = nn.Parameter(torch.randn(input_shape[1], 1).uniform_())

    def forward(self, inputs):
        # Embedding using Time2Vec
        x = inputs
        original = self.w * x + self.p
        x = x.float()
        original = original.float()
        tdot = torch.tensordot(x, self.W, dims=1)
        tdot = tdot + self.P
        sin_trans = torch.sin(tdot)

        x = torch.cat([sin_trans, original], -1)
        return x
