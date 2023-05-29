import torch.nn as nn

from Order_evaluation.MLP import MLP
from Order_evaluation.Time2VecEmbed import Embed
from Order_evaluation.autocorrelation_layer import autocorrelation_layer
from Order_evaluation.feed_forward import FeedForward
from Order_evaluation.series_decomposition import series_decomposition_block


class autoformer_decoder(nn.Module):
    def __init__(self, c: int, window: int, prediction_len: int, num_layers: int,
                 num_heads: int, trainable: bool, d_model: int, decoders: int):
        super(autoformer_decoder, self).__init__()
        self.c = c
        self.window = window
        self.d_model = d_model
        self.num_heads = num_heads
        self.trainable = trainable
        self.prediction_len = prediction_len
        self.num_layers = num_layers
        self.decoders = decoders

        self.autocorrelation1 = autocorrelation_layer(c=self.c,
                                                      output=self.prediction_len, heads=self.num_heads,
                                                      trainable=self.trainable, d_model=self.d_model)

        self.autocorrelation2 = autocorrelation_layer(c=self.c,
                                                      output=self.prediction_len, heads=self.num_heads,
                                                      trainable=self.trainable, d_model=self.d_model)

        self.series_decomp1 = series_decomposition_block(window=int(self.window / 2 + self.prediction_len),
                                                         num_heads=self.d_model, return_trend=True)
        self.series_decomp2 = series_decomposition_block(window=int(self.window / 2 + self.prediction_len),
                                                         num_heads=self.d_model, return_trend=True)
        self.series_decomp3 = series_decomposition_block(window=int(self.window / 2 + self.prediction_len),
                                                         num_heads=self.d_model, return_trend=True)
        self.feed_forward = FeedForward(d_model=self.d_model, dropout=0.1, dim_ff=self.d_model * 4)
        self.MLP = nn.ModuleList([MLP(d_model=self.d_model, out_length=6, num_layers=self.num_layers) for
                                  _ in range(4)])
        self.embed = Embed(self.d_model, self.window)

    def forward(self, x_de, x_en, x_mean):
        self.embed.build(x_de.shape)
        x_de = self.embed(x_de)

        for _ in range(self.decoders):
            x_des = self.autocorrelation1(x_de)
            x_des += x_de

            s_de, t_de = self.series_decomp1(x_des)

            x_des_2 = self.autocorrelation2((x_de, x_en))
            x_des = x_des_2 + s_de

            s_de, t_de_1 = self.series_decomp2(x_des)

            x_des = self.feed_forward(s_de)
            x_des += s_de

            x_de, t_de_2 = self.series_decomp3(x_des)

            t_de = self.MLP[0](t_de)
            t_de_1 = self.MLP[1](t_de_1)
            t_de_2 = self.MLP[2](t_de_2)
            x_mean = x_mean + t_de + t_de_1 + t_de_2

        x_de = self.MLP[3](x_de)

        return x_de, x_mean
