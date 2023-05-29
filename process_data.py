import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from dataset import make_ts_DataLoaders, make_order_DataLoaders
from import_data import load_all_tables

Minmax_f = MinMaxScaler()
Minmax_s = MinMaxScaler()

replace_f1 = {'BID': 1, 'ASK': 2, np.nan: 0, 0: 0}
replace_f2 = {'ORDER': 1, 'CANCEL': 2, 'TRADE': 3, np.nan: 0, 0: 0}
replace_f3 = {True: 1, False: 0}
replace_s = {'CANCEL': -1, 'PART_CANCEL': 0, 'PART_TRADE': 1, 'ORDER': 2, 'TRADE': 3, }


def load(state_path: str or None, flow_path: str or None, firm: str, last_split: int = 0, to_split: int = 10):
    state_files = [file for file in listdir(state_path) if isfile(join(state_path, file))]
    flow_files = [file for file in listdir(flow_path) if isfile(join(flow_path, file))]

    states_split_start = int(len(state_files) * last_split / to_split)
    flows_split_start = int(len(flow_files) * last_split / to_split)

    states_split_end = int(len(state_files) / to_split) + states_split_start
    flows_split_end = int(len(flow_files) / to_split) + flows_split_start

    state_files = state_files[states_split_start:states_split_end]
    flow_files = flow_files[flows_split_start:flows_split_end]

    states_list = []
    flows_list = []
    for i, (states_p, flow_p) in enumerate(zip(state_files, flow_files)):
        fl = load_all_tables(flow_path + "/" + flow_p)
        for key in fl.keys():
            if "ORDERS" in key:
                flow = fl[key]
                break
        flow = flow[['STEP_CREATED', 'TIME_CREATED', 'SIDE', 'UPDATE', 'QUANTITY', 'DELTA_QUANTITY',
                     'TRADED', 'QUANTITY_TO_TRADE']]
        flow.loc[:, 'TIME_CREATED'] = flow['TIME_CREATED'].apply(lambda x: x + i * 86400000)
        flows_list.append(flow)

        state = pd.read_hdf(state_path + "/" + states_p)
        state.loc[:, 'TIME_CREATED'] = state['TIME_CREATED'].apply(lambda x: x + i * 86400000)
        states_list.append(state)

    states = pd.concat(states_list, ignore_index=True)
    flows = pd.concat(flows_list, ignore_index=True)

    states.sort_values(by='TIME_CREATED', inplace=True)
    flows.sort_values(by='TIME_CREATED', inplace=True)

    states['UPDATE_TYPE'] = states['UPDATE_TYPE'].map(replace_s)
    states['MIDPRICE'] = (states['ASK_1_PRICE'] + states['BID_1_PRICE']) / 2

    ask_quantity_columns = [f'ASK_{i}_QUANTITY' for i in range(1, 11)]
    bid_quantity_columns = [f'BID_{i}_QUANTITY' for i in range(1, 11)]

    states['SUPPLY'] = states[ask_quantity_columns].sum(axis=1) / 10
    states['DEMAND'] = states[bid_quantity_columns].sum(axis=1) / 10

    drop_columns = [f'ASK_{i}_PRICE' for i in range(1, 11)] + [f'ASK_{i}_QUANTITY' for i in range(1, 11)] + \
                   [f'BID_{i}_PRICE' for i in range(1, 11)] + [f'BID_{i}_QUANTITY' for i in range(1, 11)]
    states.drop(columns=drop_columns, inplace=True)

    states[['MIDPRICE', 'SUPPLY', 'DEMAND']] = Minmax_s.fit_transform(states[['MIDPRICE', 'SUPPLY', 'DEMAND']])

    flows.fillna(0, inplace=True)
    flows['SIDE'] = flows['SIDE'].map(replace_f1)
    flows['UPDATE'] = flows['UPDATE'].map(replace_f2)
    flows['TRADED'] = flows['TRADED'].map(replace_f3)
    flows[['QUANTITY', 'DELTA_QUANTITY', 'QUANTITY_TO_TRADE']] = Minmax_f.fit_transform(
        flows[['QUANTITY', 'DELTA_QUANTITY', 'QUANTITY_TO_TRADE']])

    flows.set_index(keys='TIME_CREATED', inplace=True)
    states.set_index(keys='TIME_CREATED', inplace=True)
    l_states = states.shape[0]

    states = states.loc[:, ['MIDPRICE', 'SUPPLY', 'DEMAND', 'UPDATE_TYPE', 'STEP']]

    trainoe_f = flows
    traints_s, trainoe_s = states.iloc[int(l_states * 0.5):], states.iloc[:int(l_states * 0.5)]

    traints_s.to_parquet(f'D:/loaded_data/timeseries/{firm}_{last_split}_states.parquet.gzip', compression='gzip')

    trainoe_s.to_parquet(f'D:/loaded_data/order_eval/{firm}_{last_split}_states.parquet.gzip', compression='gzip')
    trainoe_f.to_parquet(f'D:/loaded_data/order_eval/{firm}_{last_split}_flow.parquet.gzip', compression='gzip')

    return traints_s, trainoe_f, trainoe_s


def preprocess(statedf: pd.DataFrame, flowdf: pd.DataFrame, window: int, horizon: int, path: str,
               resolution: int or None = None):
    if resolution is not None:
        statedf = statedf.rolling(window=resolution).mean()

    # Shift statedf to get previous and next statedf for each column
    for column in statedf.columns:
        if column != 'STEP':
            for i in range(1, window + 1):
                statedf[f'previous_{column}_{i}'] = statedf[column].shift(i)
        if column == "MIDPRICE":
            for i in range(1, horizon + 1):
                statedf[f'next_{column}_{i}'] = statedf[column].shift(-i)

    statedf = statedf.dropna()

    statedf.sort_values('TIME_CREATED', inplace=True)

    # Join orders and statedf on 'TIME_CREATED'
    merged = statedf.merge(flowdf, how='outer', left_index=True, right_index=True)

    merged.dropna(inplace=True)

    merged.to_parquet(path)
    return merged


def make_batches(horizon: int, window: int, state_path: str or None, flow_path: str or None,
                 shuffle_batches: bool = False, batches: int = 32, resolution: int or None = None):
    # A generator for getting all the data. Will yield all of the data for each of the firms, firm by firm.
    x_oe, y_oe, x_ts, y_ts, orders = [], [], [], [], []
    firms = listdir(state_path)
    to_split = 10

    for i, firm in enumerate(firms):
        for split in range(to_split):
            ts_state_file = f'D:/loaded_data/timeseries/{firm}_{split}_states.parquet.gzip'
            oe_flow_file = f'D:/loaded_data/order_eval/{firm}_{split}_flow.parquet.gzip'
            oe_state_file = f'D:/loaded_data/order_eval/{firm}_{split}_states.parquet.gzip'
            oe = f'D:/preprocessed_data/{firm}_{split}_merged_w{window}_h{horizon}_resolution_{resolution}.parquet.gzip'
            ts = f'D:/preprocessed_data/{firm}_{split}_ts_w{window}_h{horizon}_resolution_{resolution}.parquet.gzip'

            if not os.path.exists(ts_state_file) or not os.path.exists(oe_flow_file) or not \
                    os.path.exists(oe_state_file):
                order_states = state_path + firm
                order_flows = flow_path + firm
                ts_statedf, oe_flowdf, oe_statedf = load(firm=firm, state_path=order_states, flow_path=order_flows,
                                                         last_split=split, to_split=to_split)

            else:
                ts_statedf = pd.read_parquet(ts_state_file)
                oe_flowdf = pd.read_parquet(oe_flow_file)
                oe_statedf = pd.read_parquet(oe_state_file)

            if os.path.exists(oe):
                merged = pd.read_parquet(oe)
            else:
                merged = preprocess(oe_statedf, oe_flowdf, window, horizon, oe, resolution)

            if os.path.exists(ts):
                timeseries_df = pd.read_parquet(ts)
            else:
                timeseries_df = get_ts(window=window, horizon=horizon, df=ts_statedf, path_=ts, resolution=resolution)

            # Get the previous and next columns for both timeseries and orders
            previous_cols = [col for col in merged.columns if 'previous' in col]
            next_cols = [col for col in merged.columns if 'next' in col]

            prev = torch.tensor(merged[previous_cols].values.astype(np.float32), dtype=torch.float32)
            next = torch.tensor(merged[next_cols].values.astype(np.float32), dtype=torch.float32)
            order = torch.tensor(merged.iloc[:, -7:].values.astype(np.float32), dtype=torch.float32)

            x_oe.append(prev)
            y_oe.append(next)
            orders.append(order)

            ts_prev = [col for col in timeseries_df.columns if 'previous' in col]
            ts_next = [col for col in timeseries_df.columns if 'next' in col]

            timeseries_x = torch.tensor(timeseries_df[ts_prev].values.astype(np.float32), dtype=torch.float32)
            timeseries_y = torch.tensor(timeseries_df[ts_next].values.astype(np.float32), dtype=torch.float32)
            x_ts.append(timeseries_x)
            y_ts.append(timeseries_y)

            # For time series: unsqueeze at 2 so the shape will be (batch, columns, 1)
            oe_x = torch.cat(x_oe, dim=0).unsqueeze(2)
            oe_y = torch.cat(y_oe, dim=0).unsqueeze(2)
            # For orders: unsqueeze at 1 so the shape will be (batch, 1, columns)
            orders = torch.cat(orders, dim=0).unsqueeze(1)

            ts_x = torch.cat(x_ts, dim=0).unsqueeze(2)
            ts_y = torch.cat(y_ts, dim=0).unsqueeze(2)

            ts_dataloader, _ = make_ts_DataLoaders(x=ts_x, y=ts_y, batch_size=batches, shuffle=shuffle_batches)
            oe_dataloader, _ = make_order_DataLoaders(x=oe_x, y=oe_y, orders=orders, batch_size=batches,
                                                      shuffle=shuffle_batches)
            yield ts_dataloader, oe_dataloader


def get_ts(window: int, horizon: int, df: pd.DataFrame, path_: str, resolution: int or None = None):
    if resolution is not None:
        df = df.rolling(window=resolution).mean()

    for column in df.columns:
        if column != 'STEP':
            for i in range(1, window + 1):
                df[f'previous_{column}_{i}'] = df[column].shift(i)
        if column == "MIDPRICE":
            for i in range(1, horizon + 1):
                df[f'next_{column}_{i}'] = df[column].shift(-i)
    df.dropna(inplace=True)
    df.to_parquet(path=path_)

    return df
