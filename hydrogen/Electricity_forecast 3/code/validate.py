from NNs import LSTM
from xgb_model import  walk_forward_forecast
from data_preparation import  get_lmp_full_dl_map
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

if __name__ == "__main__":

    train_ratio = 0.7
    n_in = 24 * 4
    n_out = 1
    nn_path = "models/LSTM_model"
    xgb_path = "models/model.json"
    real_time_path = "data/real_time_lmp"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lmp_ts_dict_test = get_lmp_full_dl_map(real_time_path, n_in=n_in)

    mapes = []
    maes = []

    for key in lmp_ts_dict_test.keys():
        nn_model = LSTM(input_dim=1, hidden_dim=128, num_layers=1, output_dim=n_out)
        nn_model.load_state_dict(torch.load(f"models/LSTM_model_{key[-7:]}"))
        nn_model.to(device).double()
        nn_model.eval()

        x, y = lmp_ts_dict_test[key][0], lmp_ts_dict_test[key][1]
        x_test, y_test = x[-24:], y[-24:]
        nn_predictions = walk_forward_forecast(
            model=nn_model,
            data=x_test,
            horizon=len(y_test),
            torch_=True
        )

        try:
            mae = mean_absolute_error(y_test, nn_predictions)
            mape = mean_absolute_percentage_error(y_test, nn_predictions)
            mapes.append(mape)
            maes.append(mae)
            x_coord = [i for i in range(len(x))]
            plt.figure(figsize=(16, 10))
            plt.title(f"name: {key}, mape: {mape}, mae: {mae}")
            plt.plot(x_coord[-2*len(y_test):], np.concatenate([x[-len(y_test):, -1], y_test]), label="real values")
            print(nn_predictions.shape)
            plt.plot(x_coord[-len(y_test):], nn_predictions, label="nn predicted values")
            #plt.plot(x_coord[-len(y):], xgb_predictions, label="xgb predicted values")
            plt.grid()
            plt.legend()
            plt.savefig(f"figures/{key[-7:]}.png")

        except IndexError:
            pass

    print("mean MAPE: ")
    print(np.mean(np.array(mapes)))
    print("mean MAE:")
    print(np.mean(np.array(maes)))
