import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model: int, out_length: int, num_layers: int):
        super(MLP, self).__init__()
        self.d_model = d_model
        self.out_length = out_length
        self.num_layers = num_layers
        self.in_layer = nn.Linear(self.d_model, self.out_length)
        self.layers = nn.ModuleList([nn.Linear(self.out_length, self.out_length) for _ in range(num_layers)])

    def forward(self, inputs):
        x = inputs
        x = self.in_layer(x)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
        return x

