from torch import nn

from Order_evaluation.add_and_norm_layer import Norm
from Order_evaluation.multihead_attention_layer import MultiHeadAttention
from Order_evaluation.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float, eps: float):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.eps = eps
        self.dim_ff = dim_ff

        self.norm1 = Norm(d_model=self.d_model, eps=eps)
        self.norm2 = Norm(d_model=self.d_model, eps=eps)

        self.attention = MultiHeadAttention(n_heads=self.n_heads, d_model=self.d_model, dropout=self.dropout)
        self.feedforward = FeedForward(d_model=self.d_model, dropout=self.dropout, dim_ff=dim_ff)

        self.dropout1 = nn.Dropout(p=self.dropout)
        self.dropout2 = nn.Dropout(p=self.dropout)

    def forward(self, x):
        temp = self.norm1(x)
        x = x + self.dropout1(self.attention(temp, temp, temp))
        temp = self.norm2(x)
        x = x + self.dropout2(self.feedforward(temp))
        return x
