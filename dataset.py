import torch
from torch.utils.data import Dataset, DataLoader, random_split

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class TimeSeriesWithOrderDataset(Dataset):
    def __init__(self, x, y, order):
        self.x = x
        self.y = y
        self.order = order

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.order[idx]

def make_ts_DataLoaders(x, y, batch_size: int = 32, shuffle: bool = False):

    timeseries_dataset = TimeSeriesDataset(x, y)

    train_size = int(0.8 * len(timeseries_dataset))
    val_size = len(timeseries_dataset) - train_size


    train, validate = random_split(timeseries_dataset, [train_size, val_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(validate, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader

def make_order_DataLoaders(x, y, orders, batch_size: int = 32, shuffle: bool = False):

    order_dataset = TimeSeriesWithOrderDataset(x, y, orders)
    train_size = int(0.8 * len(order_dataset))
    val_size = len(order_dataset) - train_size

    train, validate = random_split(order_dataset, [train_size, val_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(validate, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader




