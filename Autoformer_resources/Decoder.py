import torch.nn as nn

from Autoformer_resources.Embed import Embed
from Autoformer_resources.autocorrelation_layer import autocorrelation_layer
from Autoformer_resources.feed_forward import FeedForward
from Autoformer_resources.series_decomposition import series_decomposition_block


class autoformer_decoder(nn.Module):
    def __init__(self, lags: int, c: int, window: int, prediction_len: int,
                 input_len: int, num_heads: int, trainable: bool, d_model: int):
        super(autoformer_decoder, self).__init__()
        self.lags = lags
        self.c = c
        self.input_len = input_len
        self.d_model = d_model
        self.window = window
        self.num_heads = num_heads
        self.trainable = trainable
        self.prediction_len = prediction_len
        self.autocorrelation = nn.ModuleList([autocorrelation_layer(lags=self.lags, c=self.c, use_multiple=False,
                                                                    output=self.prediction_len, heads=self.num_heads,
                                                                    trainable=self.trainable, d_model=self.d_model),
                                              autocorrelation_layer(lags=self.lags, c=self.c, use_multiple=True,
                                                                    output=self.prediction_len, heads=self.num_heads,
                                                                    trainable=self.trainable, d_model=self.d_model)])
        self.series_decomp = nn.ModuleList(
            [series_decomposition_block(window=int(self.window / 2 + self.prediction_len),
                                        num_heads=self.d_model, return_trend=True) for _ in range(4)])
        self.feed_forward = FeedForward(d_model=self.d_model, dropout=0.1, dim_ff=self.d_model * 4)
        self.embed = Embed(self.prediction_len, self.d_model, self.input_len)

    def forward(self, x_de, x_en):
        x_des = self.autocorrelation[0](x_de)
        x_des += x_de
        s_de, t_de = self.series_decomp[0](x_des)

        x_des_2 = self.autocorrelation[1]((x_de, x_en))
        x_des = x_des_2 + s_de
        s_de, t_de_1 = self.series_decomp[1](x_des)

        x_des = self.feed_forward(s_de)
        x_des += s_de
        s_de, t_de_2 = self.series_decomp[2](x_des)
        return s_de, t_de, t_de_1, t_de_2
