import torch.nn as nn

from Autoformer_resources.Embed import Embed
from Autoformer_resources.autocorrelation_layer import autocorrelation_layer
from Autoformer_resources.encoder_series_decomposition import enc_series_decomposition_block
from Autoformer_resources.feed_forward import FeedForward


class autoformer_encoder(nn.Module):
    def __init__(self, lags: int, c: int, window: int, prediction_len: int, num_heads: int,
                 trainable: bool, d_model: int):
        super(autoformer_encoder, self).__init__()
        self.lags = lags
        self.c = c
        self.num_heads = num_heads
        self.trainable = trainable
        self.d_model = d_model
        self.window = window
        self.prediction_len = prediction_len
        self.embed = Embed(horizon=self.prediction_len, d_model=self.d_model, input_length=self.window)
        self.autocorrelation = autocorrelation_layer(lags=self.lags, c=self.c, use_multiple=False,
                                                     output=self.prediction_len, heads=self.num_heads,
                                                     trainable=self.trainable, d_model=self.d_model)
        self.series_decomp = nn.ModuleList(
            [enc_series_decomposition_block(window=int(self.window / 2 + self.prediction_len),
                                            num_heads=self.d_model) for _ in range(2)])
        self.feed_forward = FeedForward(d_model=self.d_model, dim_ff=self.d_model * 4, dropout=0.1)

    def forward(self, x_in):
        # Get autocorrelation
        s_en = self.autocorrelation(x_in)
        # Add values
        s_en += x_in
        # Series decomposition, get rid of trend part
        s_en = self.series_decomp[0](s_en)
        # Feed forward net
        t = s_en.clone()
        t = self.feed_forward(t)
        s_en += t
        s_en = self.series_decomp[1](s_en)
        return s_en
