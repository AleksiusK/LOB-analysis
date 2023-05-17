#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:53:22 2022

@author: kanniain
"""

import pandas as pd
from typing import Dict


# df = pd.read_hdf('AAPL_states_52_62.h5', '/STATES_DATA')

# %% Flow data

def load_all_tables(path: str) -> Dict[str, pd.DataFrame]:
    store = pd.HDFStore(path)
    store.keys()
    table_names = store.keys()
    store.close()

    df = dict()

    for table_name in table_names:
        df[table_name] = pd.read_hdf(path, table_name)
    return df

# FlowData = load_all_tables('flow/AAPL/14S040814-v50-AAPL_OCT2.h5')
# %% state data (every second)

# StateData = pd.read_hdf('states/secondstates/AAPL/14S040814-v50-AAPL_OCT2.h5')


# df5 = pd.read_hdf('AAPL_14S040814-v50-AAPL_OCT2.h5')