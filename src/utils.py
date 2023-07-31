import os
import pickle
import fasttext
import fasttext.util
import gensim.downloader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import torch.optim as optim

from torch import nn
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from math import sqrt


def check_adf(y):
    adfinput = adfuller(y)
    adftest = pd.Series(adfinput[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
    adftest = round(adftest, 4)

    for key, value in adfinput[4].items():
        adftest["Critical Value (%s)" % key] = value.round(4)
    return adftest


def check_stationarity(y, lags_plots=48, figsize=(22, 8), title='Trend'):
    # Creating plots of the DF
    y = pd.Series(y)
    fig = plt.figure()

    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (1, 1))
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=2)

    lags_plots = min(len(y) // 2 - 1, lags_plots)

    y.plot(ax=ax1, figsize=figsize)
    ax1.set_title(title)
    plot_acf(y, lags=lags_plots, zero=False, ax=ax2)
    plot_pacf(y, lags=lags_plots, zero=False, ax=ax3)
    sns.distplot(y, bins=int(sqrt(len(y))), ax=ax4)
    ax4.set_title('Distribution Chart')

    plt.tight_layout()

    adftest = check_adf(y)

    # print(adftest)

    if adftest[0].round(2) < adftest[5].round(2):
        print('\nThe Test Statistics is lower than the Critical Value of 5%.\nThe series seems to be stationary')
    else:
        print("\nWarning: The Test Statistics is higher than the Critical Value of 5%.\nThe series isn't stationary")
    plt.show()


def get_scaler(scaler_type):
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = FunctionTransformer()
    return scaler


def get_sample_weight(start, end, n, mode='linear'):
    if mode == 'linear':
        return np.linspace(start, end, n)
    else:
        raise NotImplementedError


def load_word2vec_embeddings(country_list, cache_path='./data/pretrained_embedding.pkl'):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    else:
        emb_dict = {}
        word2vec = gensim.downloader.load('word2vec-google-news-300')
        for c in country_list:
            cw = c
            if c == 'Korea (ROK)':
                cw = 'SouthKorea'
            elif c == 'Chinese Taipei':
                cw = 'Taiwan'
            elif c == 'Hong Kong SAR':
                cw = 'HongKong'
            elif c == 'Macao, China':
                cw = 'Macao'
            elif c == 'New Zealand':
                cw = 'NewZealand'
            emb_dict[c] = word2vec[cw]
        with open(cache_path, 'wb') as f:
            pickle.dump(emb_dict, f)
        return emb_dict


def load_fasttext_embeddings(country_list, dimension, cache_path='./fasttext_cache'):
    os.makedirs(cache_path, exist_ok=True)
    cache_path = os.path.join(cache_path, f'{dimension}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    else:
        emb_dict = {}
        fasttext.util.download_model('en', if_exists='ignore')
        ft = fasttext.load_model('cc.en.300.bin')
        fasttext.util.reduce_model(ft, dimension)
        for c in country_list:
            cw = c
            if c == 'Korea (ROK)':
                cw = 'South Korea'
            elif c == 'Chinese Taipei':
                cw = 'Taiwan'
            elif c == 'Hong Kong SAR':
                cw = 'Hong Kong'
            elif c == 'Macao, China':
                cw = 'Macao'
            emb_dict[c] = ft.get_word_vector(cw)
        with open(cache_path, 'wb') as f:
            pickle.dump(emb_dict, f)
        return emb_dict


def dummy_lagged_df(train_y, train_x=None, lag=5, dropna=True, col_name='Target'):
    col_added = []
    if train_x is None:
        train_x = pd.DataFrame()
    train_y = train_y.copy()
    train_x = train_x.copy()
    for i in range(lag, 0, -1):
        train_x[col_name+'Lag'+str(i)] = train_y.shift(i)
        col_added.append(col_name+'Lag'+str(i))
    if dropna:
        na_idx = train_x.notna().all(axis=1)
        train_x = train_x[na_idx]
        train_y = train_y[na_idx]
    return train_y, train_x, col_added


def get_torch_optimizer(optim_type, model_parameters, lr, **kwargs):
    if optim_type == 'sgd':
        return optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=0.001, **kwargs)
    elif optim_type == 'adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=0.001, **kwargs)
    elif optim_type == 'nadam':
        return optim.NAdam(model_parameters, lr=lr, weight_decay=0.001, **kwargs)
    elif optim_type == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=lr, weight_decay=0.001, momentum=0.9, **kwargs)
    else:
        raise NotImplementedError


def mase_loss(output, target, scale):
    return (nn.functional.l1_loss(output, target, reduction='none') / scale).mean()


def mean_absolute_scaled_error(target, output, scale):
    return np.mean(np.abs(output - target) / scale)


def compute_mase_scale(target, m=12, h=12):
    if isinstance(target, pd.Series):
        target = target.values
    sums = []
    for t in range(m, len(target) - h):
        sums.append(abs(target[t] - target[t-m]))
    return np.mean(sums)


def get_torch_criterion(criterion_type):
    if criterion_type == 'mse':
        return nn.MSELoss()
    elif criterion_type in ('l1', 'mae'):
        return nn.L1Loss()
    elif criterion_type == 'mase':
        return mase_loss
    else:
        raise NotImplementedError


def inverse_target(target, seasonal_component, scaler):
    target = np.asarray(target)
    if target.ndim == 1:
        target = np.expand_dims(target, -1)
    return np.expm1(scaler.inverse_transform(target).flatten() + seasonal_component)


def get_seasonal_component(seasonal_res, target_length):
    if len(seasonal_res.seasonal) >= target_length:
        return seasonal_res.seasonal.iloc[:target_length]
    seasonal_component = pd.concat([seasonal_res.seasonal,
                                    pd.Series(np.resize(seasonal_res.seasonal.values[-12:], target_length - len(seasonal_res.seasonal)))], ignore_index=True)
    return seasonal_component


def winkler_score(lower, upper, actual, alpha=0.2):
    assert lower.shape == upper.shape == actual.shape
    width = upper - lower
    winkler = np.where((lower <= actual) & (actual <= upper), width,
                       np.where(lower > actual, width + 2 * (lower - actual) / alpha,
                                width + 2 * (actual - upper) / alpha))
    return np.mean(winkler)


def get_confidence_interval(sample, confidence=0.8):
    n = len(sample)
    se = scipy.stats.sem(sample, axis=0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h
