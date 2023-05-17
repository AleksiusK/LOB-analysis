import getopt
import logging
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import torch

from Autoformer_resources.autoformer_model import Autoformer
from Order_evaluation.Order_eval import order_evaluation
from process_data import make_batches
from setup_model import setup_af, order_eval

mpl.use('TkAgg')
window = 100
horizon = 100

logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def train_all(epochs: int, window: int, horizon: int, states: str, flows: str, patience: int):
    autoformer, optimizer, loss, epoch, start = setup_af(window, horizon)
    autoformer = autoformer.float()
    af_loss_function = torch.nn.MSELoss()

    order_evaluation, optimizer, loss, epoch, start = order_eval(window, horizon)
    order_evaluation = order_evaluation.float()
    oe_loss_function = torch.nn.MSELoss()

    try:
        ts_dataloaders, oe_dataloaders = make_batches(
            horizon=horizon,
            window=window,
            state_path=states + "/",
            flow_path=flows + "/")
        logging.info("Batches created. Starting training.")
        losses = train_forecast_model(model=autoformer, dataloader=ts_dataloaders, epochs=epochs,
                                      optimizer=optimizer, loss_function=af_loss_function, patience=patience)

        plt.plot(losses)
        plt.yscale('log')
        plt.ylabel('Autoformer loss')
        plt.xlabel('Epoch')
        plt.show()

        logging.info("Autoformer trained.")

        losses = train_oe_model(model=order_evaluation, forecast=autoformer, dataloader=oe_dataloaders,
                                epochs=epochs,
                                optimizer=optimizer, loss_function=oe_loss_function, patience=patience)

        plt.plot(losses)
        plt.yscale('log')
        plt.ylabel('Order evaluation loss')
        plt.xlabel('Epoch')
        plt.show()

    except Exception as e:
        logging.error(f"Error while training. Error message: {str(e)}")
        raise

    logging.info(f"All training done. Congrats!")


def train_forecast_model(model: Autoformer, dataloader, epochs: int, optimizer, loss_function, patience: int):
    # Define device, 'cuda' if GPU is available else 'cpu'
    bar = progressbar.ProgressBar(maxval=epochs * len(dataloader),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)

    # Early stopping initialization
    epochs_no_improve = 0
    min_val_loss = np.inf

    model.train()
    torch.autograd.set_detect_anomaly(True)
    losses = []
    checkpoint_path = f'model_checkpoints/autoformer_checkpoint'  # Unique path for each model
    bar.start()
    i = 0
    for epoch in range(epochs):
        for batch in dataloader:
            bar.update(i + 1)
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

            # Early stopping check
            if loss.item() < min_val_loss:
                min_val_loss = loss.item()
                epochs_no_improve = 0
                # Save best model
                model.save_checkpoint(path=checkpoint_path, optimizer=optimizer, epoch=epoch, loss=min_val_loss)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f'Early stopping at epoch {epoch}, best loss was {min_val_loss}')
                    bar.finish()
                    return losses
            logging.info(f"Loss for batch {i} in epoch {epoch}: {loss.item()}")
            i += 1

        logging.info(f"Loss for epoch {epoch}: {np.min(losses)}")

    logging.info(f"Training model done.")
    bar.finish()
    return losses


def train_oe_model(model: order_evaluation, forecast: Autoformer, dataloader, epochs: int, optimizer, loss_function,
                   patience: int):
    # Define device, 'cuda' if GPU is available else
    bar = progressbar.ProgressBar(maxval=epochs * len(dataloader),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)
    forecast = forecast.to(device)

    # Early stopping initialization
    epochs_no_improve = 0
    min_val_loss = np.inf

    model.train()
    torch.autograd.set_detect_anomaly(True)
    losses = []
    checkpoint_path = f'model_checkpoints/order_evaluation_checkpoint'  # Unique path for each model
    bar.start()
    i = 0
    for epoch in range(epochs):
        for batch in dataloader:
            bar.update(i + 1)
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
            losses.append(loss.item())  # Update weights

            # Early stopping check
            if loss.item() < min_val_loss:
                min_val_loss = loss.item()
                epochs_no_improve = 0
                # Save best model
                model.save_checkpoint(path=checkpoint_path, optimizer=optimizer, epoch=epoch, loss=min_val_loss)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info(f'Early stopping at epoch {epoch}, best loss was {min_val_loss}')
                    bar.finish()
                    return losses
            logging.info(f"Loss for batch {i} in epoch {epoch}: {loss.item()}")
            i += 1

        logging.info(f"Loss for epoch {epoch}: {np.min(losses)}")

    logging.info(f"Training model done.")
    bar.finish()
    return losses


def main(argv):
    epochs = 100
    window = 60  # 3 hours in milliseconds = 300000
    horizon = 60  # These need to be dividable by one another
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
