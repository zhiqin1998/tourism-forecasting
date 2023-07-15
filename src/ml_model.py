import xgboost as xgb
import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from src.utils import get_scaler, dummy_lagged_df


def get_model(model_type):
    if model_type == 'lr':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = RidgeCV(alphas=(0.1, 0.25, 0.5, 1.0, 5.0, 10.0))
    elif model_type == 'elastic':
        model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.])
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=10)
    elif model_type == 'svr':
        model = LinearSVR()
    elif model_type == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=2)
    elif model_type == 'xgb':
        model = xgb.XGBRegressor()
    else:
        raise NotImplementedError
    return model


def grid_search_ml(ori_train_x, ori_train_y, score_function=None, model_types=None, preprocess=None, sample_weight=None, additional_params=None):
    if model_types is None:
        model_types = ['lr', 'ridge', 'elastic', 'mlp', 'rf', 'svr', 'xgb']
    if preprocess is None:
        preprocess = ['nopreprocess', 'standard', 'minmax']
    if additional_params is None:
        additional_params = {}
    if score_function is None:
        def score_function(gt, pr):
            return sqrt(mean_squared_error(gt, pr))

    best_score = np.inf
    best_param = None
    ori_train_x, ori_train_y = ori_train_x.copy(), ori_train_y.copy()
    for model_type in model_types:
    # for model_type in ['mlp', 'rf', 'svr', 'xgb']:
        model = get_model(model_type)
        for prep in preprocess:
            if model_type in ['lr', 'ridge', 'elastic', 'mlp', 'svr'] and prep == 'nopreprocess':
                continue #force preprocess for linear model
            for add_param in additional_params.get(model_type, [{}]):
                score = []
                kf = KFold()
                for train_index, test_index in kf.split(ori_train_x):
                    train_x, train_y = ori_train_x.iloc[train_index], ori_train_y.iloc[train_index]
                    test_x, test_y = ori_train_x.iloc[test_index], ori_train_y.iloc[test_index]
                    scaler = get_scaler(prep)
                    train_x = scaler.fit_transform(train_x)
                    test_x = scaler.transform(test_x)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = model.set_params(**add_param)
                        if model_type == 'mlp':
                            model.fit(train_x, train_y)
                        else:
                            model.fit(train_x, train_y, sample_weight=sample_weight)
                    pred = model.predict(test_x)
                    score.append(score_function(test_y, pred))
                score = np.mean(score)
                if score < best_score:
                    best_score = score
                    best_param = (model_type, prep, model.get_params())
    return best_param, best_score


def one_step_prediction(all_df, candidates, target, model, country_scaler, target_encoder, scaler, lag=1):
    pred_dict = {}
    for df in all_df:
        train_df = df.iloc[df[target].first_valid_index():]
        _, train_x, added_col = dummy_lagged_df(train_df[target], dropna=False, lag=lag)
        train_df = pd.concat([train_df, train_x], axis=1).reset_index(drop=True)

        train_df, test_df = train_df.iloc[:train_df[target].last_valid_index()].dropna(), train_df.iloc[train_df[
                                                                                                            target].last_valid_index() + 1:]

        temp = candidates.copy()
        temp.remove('Country')
        country = test_df['Country'].unique()[0]
        pred_dict[country] = []

        for i in range(len(test_df)):
            row = test_df.iloc[i:i + 1][candidates + added_col]

            for l in range(1, lag+1):
                if row['TargetLag' + str(l)].isna().sum() > 0:
                    assert len(pred_dict[country]) >= l
                    row.iloc[0, row.columns.get_loc('TargetLag' + str(l))] = pred_dict[country][-l]
                assert row['TargetLag' + str(l)].isna().sum() == 0

            row[temp] = country_scaler[country].transform(row[temp])
            row = target_encoder.transform(row)
            x = scaler.transform(row[candidates + added_col])
            y = model.predict(x)[0]
            pred_dict[country].append(y)
    return pred_dict
