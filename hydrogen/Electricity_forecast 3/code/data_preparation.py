import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import glob

DATE_COL = "OPR_DT"
DATETIME_COL = "INTERVALSTARTTIME_GMT"
GROUP_COL = "GROUP"
LMP_COL = "LMP_TYPE"
MW_COL = "MW"


# filter: LMP — location marginal prices
# sum over groups:

def series_to_supervised(data: np.ndarray,
                         n_in=30, n_out=1,
                         dropnan=True):
    df = pd.DataFrame(data)
    cols = []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    for i in range(0, n_out):
        cols.append(df.shift(-i))

    agg = pd.concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def get_lmp_train_dl_map(real_time_path: str, train_ratio: float = 0.7, n_in: int = 30, batch_size: int = 20):
    lmp_paths = glob.glob(real_time_path + "/lmp_*")
    lmp_ts_dict_train = {}
    for lmp_path in lmp_paths:
        x, y = concat_dataframes_and_supervised(lmp_path, n_in)
        x_train, y_train = x[:-24], y[:-24]
        x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
        ds = TensorDataset(x_train.reshape(x_train.shape[0], x_train.shape[1], 1), y_train.reshape(y_train.shape[0], 1))
        dl = DataLoader(ds, batch_size=batch_size)
        lmp_ts_dict_train[lmp_path[-7:]] = dl
    return lmp_ts_dict_train


def get_lmp_full_dl_map(real_time_path: str, train_ratio: float = 0.7, n_in: int = 30, batch_size: int = 20):
    lmp_paths = glob.glob(real_time_path + "/lmp_*")
    lmp_ts_dict_test = {}
    for lmp_path in lmp_paths:
        x, y = concat_dataframes_and_supervised(lmp_path, n_in)
        lmp_ts_dict_test[lmp_path] = (x, y)
    return lmp_ts_dict_test


def concat_dataframes_and_supervised(path: str, n_in: int) -> pd.DataFrame:
    paths = glob.glob(path + "/*.csv")
    arrs = []
    for path in paths:
        df = pd.read_csv(path)
        arr = df[MW_COL].values
        print(arr.shape)
        arrs.append(arr)
    full_ts = np.concatenate(arrs, axis=0)
    train_data = series_to_supervised(
        full_ts,
        n_in=n_in,
        n_out=1,
    )

    x_train, y_train = train_data[:, :-1], train_data[:, -1]
    return x_train, y_train


def preprocess_and_store_data(path, name="real_time"):
    data = pd.read_csv(path)
    data = data.sort_values(by=[DATETIME_COL]).reset_index().drop(columns=["index"])
    lmps = data[LMP_COL].unique()
    groups = data[GROUP_COL].unique()
    os.mkdir(f"data/{name}/")

    for lmp in lmps:
        os.mkdir(f"data/{name}/" + f"lmp_{lmp}")
        lmp_df = data[data[LMP_COL] == lmp]
        lmp_df = lmp_df[[DATE_COL, DATETIME_COL, MW_COL]].groupby(by=[DATE_COL, DATETIME_COL]).sum()
        lmp_df["OPR_DT"] = lmp_df.index.get_level_values('OPR_DT')
        dates = lmp_df.index.get_level_values('OPR_DT').unique()
        for date in dates:
            lmp_df_t = lmp_df[lmp_df[DATE_COL] == date]
            lmp_df_t.to_csv(f"data/{name}/" + f"lmp_{lmp}/" + f"{date}.csv")
    print("All data stored!")


def get_train_data_file(path, n_in=12, n_out=1):
    data = pd.read_csv(path)
    train_data = data[MW_COL].to_numpy()
    train_data = np.mean(train_data.reshape(-1, 12), axis=1)
    train_data = series_to_supervised(train_data, n_in=n_in, n_out=n_out)
    x_train, y_train = train_data[:, :-1], train_data[:, -1]
    return x_train, y_train


def get_train_data_full(path="real_time_lmp", n_in=30, n_out=1, train_ratio=0.7, torch_=False, batch_size=20):
    lmp_paths = list(sorted(glob.glob(path + "/lmp_*")))
    xs = []
    ys = []
    for lmp_path in lmp_paths:
        files_paths = glob.glob(lmp_path + "/*.csv")
        for file_path in files_paths[:int(len(files_paths) * train_ratio)]:
            x, y = get_train_data_file(file_path, n_in=n_in, n_out=n_out)
            xs.append(x)
            ys.append(y)
    full_x = np.concatenate(xs, axis=0)
    full_y = np.concatenate(ys, axis=0)
    print(full_x.shape)
    print(full_y.shape)
    if torch_:
        full_x = torch.Tensor(full_x.reshape(full_x.shape[0], full_x.shape[1], 1))
        full_y = torch.Tensor(full_y.reshape(full_y.shape[0], 1))
        ds = TensorDataset(full_x, full_y)
        dl = DataLoader(ds, batch_size=batch_size)
        return dl
    else:
        return full_x, full_y


def get_test_data_full(path, n_in=30, n_out=1, train_ratio=0.7):
    lmps_paths = list(sorted(glob.glob(path + "/lmp*")))
    xs = []
    ys = []
    lmps_tss = {}
    #print(lmps_tss)
    for lmp_path in lmps_paths:
        files_paths = glob.glob(lmp_path + "/*.csv")
        for file_path in files_paths[int(len(files_paths) * train_ratio):]:
            x, y = get_train_data_file(file_path, n_in=n_in, n_out=n_out)
            xs.append(x)
            ys.append(y)
            lmps_tss[file_path] = (x, y)

    full_x = np.concatenate(xs, axis=0)
    full_y = np.concatenate(ys, axis=0)
    return lmps_tss, full_x, full_y
