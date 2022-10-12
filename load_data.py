import pickle as pkl
import pandas as pd
import numpy as np


def detect_outliers_zscore(data, thres=3):
    ts = data.values
    mean = np.mean(ts)
    std = np.std(ts)
    return pd.Series(
        [np.nan if np.abs((i - mean) / std) > thres else i for i in ts],
        index=data.index,
    )


def load_data(input_dir):
    data = pkl.load(file=open(input_dir, "rb"))
    df = pd.DataFrame(data).reset_index()
    df.columns = ["time", "price"]
    df["date"] = df.time.apply(lambda x: x.date())
    df["processed_price"] = pd.concat(
        [detect_outliers_zscore(df[df.date == date].price) for date in df.date.unique()]
    ).fillna(method="ffill")
    return df
