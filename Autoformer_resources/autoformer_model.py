import logging

import torch
import torch.nn as nn

from Autoformer_resources.Decoder import autoformer_decoder
from Autoformer_resources.Embed import Embed
from Autoformer_resources.Encoder import autoformer_encoder
from Autoformer_resources.MLP import MLP
from Autoformer_resources.series_decomposition import series_decomposition_block

logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class Autoformer(nn.Module):
    def __init__(self, lags: int, c: int, pred_len: int, num_layers: int, num_heads: int, trainable: bool, d_model: int,
                 encoders: int, decoders: int, in_dim: int, window: int):
        super(Autoformer, self).__init__()
        self.lags = lags
        self.num_heads = num_heads
        self.trainable = trainable
        self.c = c
        self.in_dim = in_dim
        self.pred_len = pred_len
        self.d_model = d_model
        self.out_len = int(window / 2) + self.pred_len
        self.num_layers = num_layers
        self.window = lags
        self.encoders = encoders
        self.decoders = decoders
        self.embed = nn.ModuleList([
            Embed(horizon=self.pred_len, d_model=self.d_model, input_length=self.window),
            Embed(horizon=self.pred_len, d_model=self.d_model, input_length=self.window)
        ])
        self.MLP = nn.ModuleList([MLP(d_model=self.d_model, out_length=self.in_dim, num_layers=self.num_layers) for
                                  _ in range(4)])
        self.encoder = autoformer_encoder(lags=self.lags, c=self.c, window=self.window,
                                          prediction_len=self.pred_len,
                                          num_heads=self.num_heads, trainable=self.trainable, d_model=self.d_model)
        self.decoder = autoformer_decoder(lags=self.lags, c=self.c, window=self.window,
                                          prediction_len=self.pred_len, input_len=int(self.pred_len / 2),
                                          trainable=self.trainable, num_heads=self.num_heads, d_model=self.d_model)
        self.series_decomp = series_decomposition_block(window=int(self.window / 2), num_heads=self.num_heads,
                                                        return_trend=True)

    def forward(self, inputs, *args, **kwargs):
        x_in = inputs
        i_half = x_in.shape[1] // 2
        xf = x_in[:, i_half:, ]
        mean = torch.mean(xf[:, :, 0])
        x_ens, x_ent = self.series_decomp(xf)
        x0 = torch.zeros([x_in.shape[0], self.pred_len, x_in.shape[2]])
        xmean = torch.full((x_in.shape[0], self.pred_len, x_in.shape[2]), mean)
        x_des = torch.cat([x_ens, x0], dim=1)
        x_det = torch.cat([x_ent, xmean], dim=1)
        self.embed[0].build(x_in.shape)
        x_in = self.embed[0](x_in)
        for en in range(self.encoders):
            x_in = self.encoder(x_in)
        self.embed[1].build(x_des.shape)
        x_de = self.embed[1](x_des)
        t_de = x_det
        for dec in range(self.decoders):
            x_de, t_de0, t_de1, t_de2 = self.decoder(x_de, x_in)
            t_de0 = self.MLP[0](t_de0)
            t_de1 = self.MLP[1](t_de1)
            t_de2 = self.MLP[2](t_de2)
            t_de = t_de + t_de0 + t_de1 + t_de2
        x_de = self.MLP[3](x_de)
        x_pred = (x_de + t_de)[:, i_half:i_half + self.pred_len, :1]
        return x_pred

    def save_checkpoint(self, path, optimizer, epoch, loss):
        logging.info(f'Saving checkpoint to {path}')
        torch.save({
            'epoch': epoch,
            'model': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path)

    @classmethod
    def load_checkpoint(cls, model, optimizer, path):
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        loss = checkpoint['loss']
        epoch = checkpoint['epoch']

        return model, optimizer, loss, epoch
