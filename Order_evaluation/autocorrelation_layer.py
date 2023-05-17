import torch.nn as nn
from Order_evaluation.autocorrelation import autocorrelation_block

class autocorrelation_layer(nn.Module):
    def __init__(self, c: float, output: int, heads: int, trainable: bool, d_model: int):
        super(autocorrelation_layer, self).__init__()
        self.c = c
        self.output_len = output
        self.num_heads = heads
        self.trainable = trainable
        self.d_model = d_model
        self.autocorrelation_block = autocorrelation_block(c=self.c, output=self.output_len, heads=self.num_heads,
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

