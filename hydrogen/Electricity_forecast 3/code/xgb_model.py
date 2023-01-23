import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from tqdm import tqdm
import torch
import os


def xgboost_fit(trainx, trainy, save_model=False):
    params = {
        'eta': 0.2,
        'booster': "gblinear",
        'objective': 'reg:squarederror',
        'n_estimators': 1000
    }

    model = XGBRegressor(**params)
    model.fit(trainx, trainy)

    if save_model:
        model.save_model()

    return model


def walk_forward_forecast(model, data, horizon, torch_=False):
    predictions = []
    history = np.array([x for x in data])
    if torch_:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in tqdm(range(horizon)):
            next_step = torch.Tensor(history[-1])
            yhat = model(next_step.reshape(1, -1, 1).cuda())
            predictions.append(yhat.cpu().detach().item())
            history = np.vstack([history, np.append(history[-1][1:], [yhat.cpu().detach().numpy()])])
    else:
        for i in tqdm(range(horizon)):
            next_step = history[-1][1:]
            yhat = model.predict([next_step])
            predictions.append(yhat)
            history = np.vstack([history, np.append(history[-1][1:], [yhat])])

    return np.array(predictions).reshape(-1, )
