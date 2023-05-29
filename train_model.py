import getopt
import logging
import sys

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import seaborn as sns
import torch
from progressbar import ProgressBar

from Autoformer_resources.autoformer_model import Autoformer
from Order_evaluation.Order_eval import order_evaluation
from process_data import make_batches
from setup_model import setup_models

mpl.use('TkAgg')

logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
b1_color = sns.color_palette("flare").as_hex()[5]
b2_color = sns.color_palette("flare").as_hex()[1]
af_checkpoint = 'model_checkpoints/autoformer_checkpoint'
oe_checkpoint = 'model_checkpoints/order_evaluation_checkpoint'


def train_all(epochs: int, window: int, horizon: int, states: str, flows: str, patience: int):
    model_dict = setup_models(window=window, horizon=horizon, af_checkpoint=af_checkpoint,
                              oe_checkpoint=oe_checkpoint, to_train=True)

    autoformer = model_dict["Autoformer"]["model"].float()
    autoformer_optimizer = model_dict["Autoformer"]["optimizer"]
    autoformer_loss_fn = model_dict["Autoformer"]["loss function"]

    order_evaluation = model_dict["Order evaluation"]["model"].float()
    order_evaluation_optimizer = model_dict["Order evaluation"]["optimizer"]
    order_evaluation_loss_fn = model_dict["Order evaluation"]["loss function"]

    try:
        batch_generator = make_batches(
            horizon=horizon,
            window=window,
            state_path=states + "/",
            flow_path=flows + "/",
            shuffle_batches=True,
            resolution=1000)  # By applying resolution, the function will use rolling averages instead of the raw
        # data. This will affect how you should read the "window" and "horizon" parameters. For example, as the data
        # I am using is in the intervals of milliseconds, scaling by 60 000 will result in the data being in minutes
        # instead of milliseconds. Useful values for resolution might be 1000 (1s) or 60 000 (1min)

        for ts_dataloader, oe_dataloader in batch_generator:
            logging.info("Batches created. Starting training.")
            losses_af = train_forecast_model(model=autoformer, dataloader=ts_dataloader, epochs=epochs,
                                             optimizer=autoformer_optimizer, loss_function=autoformer_loss_fn,
                                             patience=patience)

            logging.info("Autoformer trained.")

            losses_oe = train_oe_model(model=order_evaluation, forecast=autoformer, dataloader=oe_dataloader,
                                       epochs=epochs,
                                       optimizer=order_evaluation_optimizer, loss_function=order_evaluation_loss_fn,
                                       patience=patience)

            logging.info("Order evaluation trained.")

            sns.barplot(x=np.arange(epochs), y=losses_af, color=b1_color)
            sns.barplot(x=np.arange(epochs), y=losses_oe, color=b2_color)

            top = mpatches.Patch(color=b1_color, label='Losses for timeseries only forecast')
            bottom = mpatches.Patch(color=b2_color, label="Losses for order and forecast combined")
            plt.legend(handles=[top, bottom])

            plt.show()

    except Exception as e:
        logging.error(f"Error while training. Error message: {str(e)}")
        raise

    logging.info(f"All training done. Congrats!")


def train_forecast_model(model: Autoformer, dataloader, epochs: int, optimizer, loss_function, patience: int):
    # Define device, 'cuda' if GPU is available else 'cpu'
    bar = ProgressBar(max_value=epochs * len(dataloader), widgets=['[', progressbar.Timer(), ']',
                                                                   progressbar.GranularBar(), ' ',
                                                                   progressbar.Percentage()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)

    # Early stopping initialization
    epochs_no_improve = 0

    model.train()
    torch.autograd.set_detect_anomaly(True)
    af_min_val_loss = np.inf
    overall_losses = []
    i = 0
    for epoch in range(epochs):
        losses = []
        for batch in dataloader:
            x, y = batch
            # Move x and y to the device
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()  # Clear gradients
            output = model(x.float())  # Forward pass
            loss = loss_function(output, y.float())
            loss.backward()  # Backward pass
            optimizer.step()  # Calculate loss
            losses.append(loss.item())  # Update weights
            i += 1
            bar.increment()

            # Early stopping check
            if loss.item() < af_min_val_loss:
                af_min_val_loss = loss.item()
                epochs_no_improve = 0
                # Save best model
                model.save_checkpoint(path=af_checkpoint, optimizer=optimizer, epoch=epoch, loss=af_min_val_loss)
            else:
                epochs_no_improve += 1
            logging.info(f"Loss for batch {i} in epoch {epoch}: {loss.item()}")

        if epochs_no_improve >= patience:
            logging.info(f'Early stopping at epoch {epoch}, best loss was {af_min_val_loss}')
            break

        logging.info(f"Best loss for epoch {epoch}: {np.min(losses)}")
        overall_losses.append(np.mean(losses))

    logging.info(f"Training model done.")
    bar.finish()
    return overall_losses


def train_oe_model(model: order_evaluation, forecast: Autoformer, dataloader, epochs: int, optimizer, loss_function,
                   patience: int):
    # Define device, 'cuda' if GPU is available else
    bar = ProgressBar(max_value=epochs * len(dataloader), widgets=['[', progressbar.Timer(), ']',
                                                                   progressbar.GranularBar(), ' ',
                                                                   progressbar.Percentage()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)
    forecast = forecast.to(device)

    # Early stopping initialization
    epochs_no_improve = 0

    model.train()
    torch.autograd.set_detect_anomaly(True)
    oe_min_val_loss = np.inf
    overall_loss = []
    i = 0
    for epoch in range(epochs):
        losses = []
        for batch in dataloader:
            x, y, order = batch
            # Move x and y to the device
            x = x.to(device)
            y = y.to(device)
            order = order.to(device)

            estimate = forecast(x.float())
            optimizer.zero_grad()  # Clear gradients
            output = model((estimate.float(), order))  # Forward pass
            loss = loss_function(output, y.float())
            loss.backward()  # Backward pass
            optimizer.step()  # Calculate loss
            losses.append(loss.item())
            i += 1
            bar.increment()  # Update weights

            # Early stopping check
            if loss.item() < oe_min_val_loss:
                oe_min_val_loss = loss.item()
                epochs_no_improve = 0
                # Save best model
                model.save_checkpoint(path=oe_checkpoint, optimizer=optimizer, epoch=epoch, loss=oe_min_val_loss)
            else:
                epochs_no_improve += 1
            logging.info(f"Loss for batch {i} in epoch {epoch}: {loss.item()}")
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping at epoch {epoch}, best loss was {oe_min_val_loss}')
        break

        logging.info(f"Best loss for epoch {epoch}: {np.min(losses)}")
        overall_loss.append(np.mean(losses))

    logging.info(f"Training model done.")
    bar.finish()
    return overall_loss


def main(argv):
    # Default arguments
    epochs = 1
    window = 10  # 3 hours in milliseconds = 300000
    horizon = 10  # These need to be dividable by one another
    patience = 80
    state = "D:/US_LOB/states/secondstates"
    flow = "D:/US_LOB/flow"
    arg_help = "{0} -e <epoch> -w <window> -h <horizon> -s <state path> -f <flow path>".format(argv[0])
    info = "Leaving the input blank means using default inputs."

    try:
        opts, args = getopt.getopt(argv[1:], "hi:e:w:h:s:f:p", ["help", "epochs=",
                                                                "window=", "horizon=", "state=", "flow="])
    except:
        print("Using default inputs.")
        return train_all(epochs=epochs, window=window, horizon=horizon, states=state, flows=flow, patience=patience)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
            print(info)
            sys.exit(2)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-w", "--window"):
            window = int(arg)
        elif opt in ("-h", "--horizon"):
            horizon = int(arg)
        elif opt in ("-s", "--state"):
            state = arg
        elif opt in ("-f", "--flow"):
            flow = arg
        elif opt in ("-p", "--patience"):
            patience = arg
        else:
            print(info)

    train_all(epochs=epochs, window=window, horizon=horizon, states=state, flows=flow, patience=patience)


if __name__ == "__main__":
    main(sys.argv)
