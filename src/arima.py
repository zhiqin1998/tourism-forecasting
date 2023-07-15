import warnings
import statsmodels.api as sm
import numpy as np
import pandas as pd

from src.utils import check_adf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from math import sqrt


# baseline arima
def grid_search_arima(ori_train_y, ori_train_x=None, score_function=None, param_grids=None):
    best_score = np.inf
    best_param = None
    ori_train_y = ori_train_y.copy()
    if ori_train_x is not None:
        ori_train_x = ori_train_x.copy()
    if param_grids is None:
        param_grids = {}
    if score_function is None:
        def score_function(gt, pr):
            return sqrt(mean_squared_error(gt, pr))

    y = pd.Series(ori_train_y)

    adftest = check_adf(y)

    stationary = adftest[0].round(2) < adftest[5].round(2)

    for p in param_grids.get('p', range(0, 2)):
        for q in param_grids.get('q', range(0, 1 if stationary else 2)):
            for r in param_grids.get('r', range(1, 4)):
                score = []
                kf = TimeSeriesSplit()
                for i, (train_index, test_index) in enumerate(kf.split(ori_train_y)):
                    if i == 0:
                        continue
                    # print(train_index, test_index)
                    if isinstance(ori_train_y, pd.Series):
                        train_y, test_y = ori_train_y.iloc[train_index], ori_train_y.iloc[test_index]
                    else:
                        train_y, test_y = ori_train_y[train_index], ori_train_y[test_index]
                    if ori_train_x is not None:
                        if isinstance(ori_train_x, pd.DataFrame):
                            train_x, test_x = ori_train_x.iloc[train_index], ori_train_x.iloc[test_index]
                        else:
                            train_x, test_x = ori_train_x[train_index], ori_train_x[test_index]
                    else:
                        train_x, test_x = None, None

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            arima_model = sm.tsa.arima.ARIMA(train_y, train_x, order=(p, q, r))
                            result = arima_model.fit()
                        except np.linalg.LinAlgError:
                            continue
                    pred = result.forecast(len(test_y), exog=test_x)
                    score.append(score_function(test_y, pred))
                score = np.mean(score)
                if score < best_score:
                    best_score = score
                    best_param = (p, q, r)
    return best_param, best_score
