import numpy as np
import torch
from torch import nn


# The normal positional encoding found in Pytorch. Coded here just for learning purposes.

class PosEncode(nn.Module):
    def __init__(self, d_model: int, order_dim: int):
        super(PosEncode, self).__init__()
        self.d_model = d_model
        self.order_dim = order_dim
        self.positional_encodings = self.create_positional_encoding()
        self.register_buffer('positional_encoding', self.positional_encodings)

    def create_positional_encoding(self):
        pos_encode = np.zeros((self.order_dim, self.d_model))
        for pos in range(self.order_dim):
            for i in range(0, self.d_model, 2):
                pos_encode[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pos_encode[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / self.d_model)))
        return torch.from_numpy(pos_encode).float()

    def forward(self, x: torch.Tensor):
        temp = self.positional_encodings[:x.size(1), :]
        temp = x + temp.unsqueeze(0)
        return temp



