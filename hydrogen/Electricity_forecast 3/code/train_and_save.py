from NNs import LSTM, train
from data_preparation import  get_lmp_train_dl_map
from torch.optim import Adam
from torch import nn
import torch
import glob


if __name__ == "__main__":

    model_path = "models"
    n_in = 24 * 4
    n_out = 1
    train_ratio = 0.7
    num_epochs_for_NN = 10
    real_time_path = "data/real_time_lmp"
    lmp_paths = glob.glob(real_time_path + "/lmp_*")
    print("Start data preparing ...")
    lmp_ts_dict_train = get_lmp_train_dl_map(real_time_path, n_in=n_in)

    print("Start training neural network")
    for key in lmp_ts_dict_train.keys():
        print(f"Start training neural network for {key}")
        neural_network = LSTM(input_dim=1, hidden_dim=128, num_layers=1, output_dim=n_out)
        neural_network = neural_network.double()
        optimizer = Adam(neural_network.parameters(), lr=0.001)
        loss_fn = nn.L1Loss()
        train(neural_network, lmp_ts_dict_train[key], num_epochs_for_NN, optimizer, loss_fn)
        print("Neural network is fitted!")

        torch.save(neural_network.state_dict(),  model_path + f"/LSTM_model_{key}")
    print("All is done!!!")