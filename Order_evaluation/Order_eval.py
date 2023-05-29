import logging

import torch
import torch.nn as nn

from Order_evaluation.Decoder import autoformer_decoder
from Order_evaluation.OE_encoder import OE_Encoder
from Order_evaluation.series_decomposition import series_decomposition_block

logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class order_evaluation(nn.Module):
    def __init__(self, window: int, c: int, pred_len: int, num_layers: int, num_heads: int,
                 trainable: bool, d_model: int, encoders: int, decoders: int, order_dim: int):
        super(order_evaluation, self).__init__()
        self.num_heads = num_heads
        self.trainable = trainable
        self.c = c
        self.order_dim = order_dim
        self.pred_len = pred_len
        self.d_model = d_model
        self.out_len = int(window / 2) + self.pred_len
        self.num_layers = num_layers
        self.window = window
        self.encoders = encoders
        self.decoders = decoders

        self.encoder = OE_Encoder(order_dim=self.order_dim, dim_ff=self.d_model * 4, d_model=self.d_model, dropout=0.1,
                                  eps=1e-9, n_heads=num_heads, n_encoders=self.encoders)

        self.decoder = autoformer_decoder(c=self.c, window=self.window,
                                          prediction_len=self.pred_len, num_layers=self.num_layers,
                                          trainable=self.trainable, num_heads=self.num_heads, d_model=self.d_model,
                                          decoders=self.decoders)
        self.series_decomp = series_decomposition_block(window=int(self.window / 2), num_heads=self.num_heads,
                                                        return_trend=True)

    def forward(self, inputs, *args, **kwargs):
        x_timeseries, x_order = inputs
        i_half = x_timeseries.shape[1] // 2

        split = x_timeseries[:, i_half:, ]
        mean = torch.mean(split[:, :, 0]).item()

        x_ens, x_ent = self.series_decomp(split)
        x0 = torch.zeros([x_timeseries.shape[0], self.pred_len, x_timeseries.shape[2]])
        x_mean = torch.full((x_timeseries.shape[0], self.pred_len, x_timeseries.shape[2]), mean)

        x_decoder = torch.cat([x_ens, x0], dim=1)
        x_mean = torch.cat([x_ent, x_mean], dim=1)

        x_order = self.encoder(x_order)

        x_decoder, x_mean = self.decoder(x_decoder, x_order, x_mean)

        x_pred = (x_decoder + x_mean)[:, i_half:i_half + self.pred_len, :1]

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
