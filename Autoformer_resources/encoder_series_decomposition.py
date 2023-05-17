import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class enc_series_decomposition_block(nn.Module):
    def __init__(self, window: int, num_heads: int):
        super(enc_series_decomposition_block, self).__init__()
        self.window = window
        self.num_heads = num_heads
        self.AVGPool = nn.AvgPool1d(kernel_size=self.window, stride=1, padding=self.window//2)

    def forward(self, input):
        x = input
        x = self.extract_trend(x) - x
        return x

    def extract_trend(self, x, window_size: int = 3):
        if window_size % 2 == 0:
            window_size += 1  # make the window size odd, to have a centered moving average

        B, C, L = x.shape  # get the dimensions of the input
        window = torch.ones(1, 1, window_size) / window_size  # create a window for convolution
        window = window.to(x.device)  # move window tensor to same device as x
        padding = window_size // 2  # calculate padding

        trend = torch.zeros_like(x)  # create an output tensor of the same size as x
        for c in range(C):  # for each channel
            trend[:, c:c + 1, :] = torch.nn.functional.conv1d(x[:, c:c + 1, :], window, padding=padding)

        return trend

