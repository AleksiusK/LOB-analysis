from os.path import exists

import numpy as np
import torch

from Autoformer_resources.autoformer_model import Autoformer as af
from Order_evaluation.Order_eval import order_evaluation

af_ckpt = 'model_checkpoints/autoformer_checkpoint'
oe_ckpt = 'model_checkpoints/order_evaluation_checkpoint'


def setup_af(window: int, horizon: int):
    if exists(af_ckpt):
        model, optimizer, loss, epoch, start = get_compiled_model("AF", window, horizon)
        return model, optimizer, loss, epoch, start
    else:
        model = af(lags=window, c=2, pred_len=horizon, num_layers=4, num_heads=8, trainable=True, d_model=512,
                   encoders=2, decoders=1, in_dim=5, window=window)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, betas=(0.9, 0.999))
        return model, optimizer, np.inf, 0, 0


def order_eval(window: int, horizon: int):
    if exists(oe_ckpt):
        model, optimizer, loss, epoch, start = get_compiled_model("OE", window, horizon)
        return model, optimizer, loss, epoch, start
    else:
        model = order_evaluation(c=2, pred_len=horizon, num_layers=4, num_heads=8, trainable=True, d_model=512,
                                 encoders=2, decoders=1, order_dim=7, window=window)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, betas=(0.9, 0.999))
        return model, optimizer, np.inf, 0, 0


def get_compiled_model(model: str, window: int, horizon: int):
    if model == "AF":
        model = af(lags=window, c=2, pred_len=horizon, num_layers=4, num_heads=8, trainable=True, d_model=512,
                   encoders=2, decoders=1, in_dim=5, window=window)
        optimizer = torch.optim.Adam(model.parameters())
        model, optimizer, loss, epoch, start = model.load_checkpoint(model=model, optimizer=optimizer,
                                                                     path=af_ckpt)
    else:
        model = order_evaluation(c=2, pred_len=horizon, num_layers=4, num_heads=8, trainable=True, d_model=512,
                                 encoders=2, decoders=1, order_dim=7, window=window)
        optimizer = torch.optim.Adam(model.parameters())
        model, optimizer, loss, epoch, start = model.load_checkpoint(model=model, optimizer=optimizer,
                                                                     path=oe_ckpt)

    return model, optimizer, loss, epoch, start
