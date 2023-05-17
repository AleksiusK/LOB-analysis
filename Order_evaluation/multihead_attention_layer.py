import torch
from torch import nn

from Order_evaluation.dotproduct_attention import DotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.head_dim = self.d_model // self.n_heads

        self.attentions = torch.nn.ModuleList(
            [DotProductAttention(d_model=self.d_model, head_dim=self.head_dim, dropout=self.dropout) for _ in range(
                self.n_heads)]
        )
        self.linear_layers = [nn.Linear(self.d_model, self.head_dim) for _ in range(3)]
        self.linear_concat = nn.Linear(self.n_heads * self.head_dim, self.d_model)
        self.linear_out = nn.Linear(self.n_heads * self.head_dim, self.d_model)



    def forward(self, query, key, value):

        query = self.linear_layers[0](query)
        key = self.linear_layers[1](key)
        value = self.linear_layers[2](value)

        output = [self_attention(query, key, value) for self_attention in self.attentions]
        output = torch.cat(output, dim=2)

        output = self.linear_concat(output)

        output = self.linear_out(output)
        return output
