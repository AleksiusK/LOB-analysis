from os.path import exists

import torch

from Autoformer_resources.autoformer_model import Autoformer as af
from Order_evaluation.Order_eval import order_evaluation


def setup_models(window: int, horizon: int, af_checkpoint: str, oe_checkpoint: str, to_train: bool = True):
    af_loss, af_epoch, oe_loss, oe_epoch = None, None, None, None
    # Define models, optimizers and loss functions.
    # On the parameters:
    
    # @param c: The multiplier used in the auto correlation function. Should be larger than 0 and smaller than 3.
    # pred_len: the length of the prediction wanted
    # num_layers: the amount of layers in the MLP
    # num_heads: the amount of heads used. Used in splitting the embedded data into batches, when forwarding to the
    # auto correlation.
    # d_model: the embedding dimension. Must be divisible by num_heads.
    # encoders: the amount of encoders to use.
    # decoders: the amount of decoders to use.
    # in_dim: the dimensionality of data on input. Example input shape: (Batch_size, window, in_dim)
    # window: the length of input data time series
    # order_dim: the amount of variables per order. Shape of order should be (Batch_size, 1, order_dim)

    autoformer = af(lags=window, c=2, pred_len=horizon, num_layers=4, num_heads=8, trainable=to_train, d_model=512,
                    encoders=2, decoders=1, in_dim=5, window=window)
    af_optim = torch.optim.Adam(autoformer.parameters(), lr=0.0008, betas=(0.9, 0.999))
    oe = order_evaluation(c=2, pred_len=horizon, num_layers=4, num_heads=8, trainable=to_train, d_model=512,
                          encoders=2, decoders=1, order_dim=7, window=window)
    oe_optim = torch.optim.Adam(oe.parameters(), lr=0.0008, betas=(0.9, 0.999))

    af_loss_fn = torch.nn.MSELoss()
    oe_loss_fn = torch.nn.MSELoss()

    if exists(af_checkpoint):
        autoformer, af_optim, af_loss, af_epoch = autoformer.load_checkpoint(model=autoformer,
                                                                             optimizer=af_optim,
                                                                             path=af_checkpoint)
    if exists(oe_checkpoint):
        oe, oe_optim, oe_loss, oe_epoch = oe.load_checkpoint(model=oe, optimizer=oe_optim,
                                                             path=oe_checkpoint)

    ret = {
        "Autoformer": {
            "model": autoformer,
            "optimizer": af_optim,
            "loss function": af_loss_fn,
            "loss": af_loss,
            "epoch": af_epoch
        },
        "Order evaluation": {
            "model": oe,
            "optimizer": oe_optim,
            "loss function": oe_loss_fn,
            "loss": oe_loss,
            "epoch": oe_epoch
        }
    }

    return ret
