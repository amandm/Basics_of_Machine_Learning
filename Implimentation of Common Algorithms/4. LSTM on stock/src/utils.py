import numpy as np
import pandas as pd

def read_data(path,debug = True):
    df = pd.read_csv(path, nrows=250 if debug else None)
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].to_numpy()
    y = np.array(df.NDX)

    return X, y

def read_stock(path,debug = True):
    df = pd.read_csv(path, nrows=250 if debug else None)
    # target =  np.array(df["Close"])
    target =  df["Close"].values
    
    return target


