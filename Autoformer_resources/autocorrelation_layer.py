import torch
import torch.nn as nn
from Autoformer_resources.autocorrelation import autocorrelation_block

class autocorrelation_layer(nn.Module):
    def __init__(self, lags: int, c: float, use_multiple: bool, output: int, heads: int, trainable: bool, d_model: int):
        super(autocorrelation_layer, self).__init__()
        self.lags = lags
        self.c = c
        self.use_multiple = use_multiple
        self.output_len = output
        self.num_heads = heads
        self.trainable = trainable
        self.d_model = d_model
        self.autocorrelation_block = autocorrelation_block(lags=self.lags, c=self.c, use_multiple=self.use_multiple,
                                                           output=self.output_len, heads=self.num_heads,
                                                           trainable=self.trainable, d_model=self.d_model)
        self.linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, inputs):
        if type(inputs) != tuple:
            q = inputs
            k = inputs
            v = inputs # Self- attention -> Q = K = V
            x = self.autocorrelation_block(q, k, v)
        else:
            q = inputs[0]
            k = self.linear(inputs[1]) # Encoder- decoder attention -> K = V
            x = self.autocorrelation_block(q, k, k)
        return x

