import torch
import torch.nn as nn

class series_decomposition_block(nn.Module):
    def __init__(self, window: int, num_heads: int, return_trend: bool):
        super(series_decomposition_block, self).__init__()
        self.window = window
        self.return_trend = return_trend
        self.num_heads = num_heads
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        xt = self.extract_trend(x)
        xs = x - xt
        if self.return_trend:
            return xt, xs
        else:
            return xs, 0

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

