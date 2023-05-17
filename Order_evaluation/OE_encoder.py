import torch.nn
from torch import nn

from Order_evaluation.encoder_layer import EncoderLayer
from Order_evaluation.order_positional_encoding import PosEncode
from Order_evaluation.embedding_layer import Embed
from Order_evaluation.add_and_norm_layer import Norm


class OE_Encoder(nn.Module):
    def __init__(self, order_dim: int, d_model: int, n_encoders: int, n_heads: int, dim_ff: int,
                 dropout: float, eps: float):
        super(OE_Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.order_dim = order_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.eps = eps

        self.embed = Embed(order_dim=self.order_dim, d_model=self.d_model)
        self.positional_encoding = PosEncode(d_model=self.d_model, order_dim=self.order_dim)
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(d_model=self.d_model, n_heads=self.n_heads,
                                                                dim_ff=self.dim_ff, dropout=self.dropout, eps=self.eps)
                                                   for _ in range(n_encoders)])
        self.norm = Norm(d_model=self.d_model, eps=self.eps)

    def forward(self, src):
        x = self.embed(src)
        x = self.positional_encoding(x)
        for _, encoder in enumerate(self.encoder_layers):
            x = encoder(x=x)
        x = self.norm(x)
        return x
