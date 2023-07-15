import warnings
import statsmodels.api as sm
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from math import sqrt


# find parameter brute force
def grid_search_sarimax(ori_train, ori_test, ori_train_x=None, ori_test_x=None, param_grids=None, score_function=None, seasonal_period=12, init_repeat=True, cv=False):
    train, test = ori_train.copy(), ori_test.copy()
    if ori_train_x is not None:
        train_x, test_x = ori_train_x.copy(), ori_test_x.copy()
    else:
        train_x, test_x = None, None
    if param_grids is None:
        param_grids = {}
    if score_function is None:
        def score_function(gt, pr):
            return sqrt(mean_squared_error(gt, pr))

    best_params = None
    best_r2 = None
    if init_repeat:
        const_prediction = np.resize(train[-seasonal_period:].values, len(test)) # use repeat prediction as baseline
        best_score = score_function(test, const_prediction)
    else:
        best_score = np.inf

    adfinput = adfuller(ori_train.dropna())
    adftest = pd.Series(adfinput[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
    adftest = round(adftest, 4)

    for key, value in adfinput[4].items():
        adftest["Critical Value (%s)" % key] = value.round(4)

    if adftest[0].round(2) < adftest[5].round(2):  # stationary
        stationary = True
    else:
        stationary = False

    if cv:
        ori_train = pd.concat([ori_train, ori_test])

    for p in param_grids.get('p', range(1, 4)):
        for q in param_grids.get('q', range(0, 1 if stationary else 2)):
            for r in param_grids.get('r', range(0, 2)):
                for P in param_grids.get('P', range(0, 2)):
                    for Q in param_grids.get('Q', range(0, 2)):
                        for R in param_grids.get('R', range(0, 2)):
                            for trend in param_grids.get('trend', ['n', 'c']):
                                if cv:
                                    score = []
                                    kf = TimeSeriesSplit(test_size=len(ori_test))
                                    for i, (train_index, test_index) in enumerate(kf.split(ori_train)):
                                        if i == 0:
                                            continue
                                        if isinstance(ori_train, pd.Series):
                                            train, test = ori_train.iloc[train_index], ori_train.iloc[test_index]
                                        else:
                                            train, test = ori_train[train_index], ori_train[test_index]
                                        if ori_train_x is not None:
                                            if isinstance(ori_train_x, pd.DataFrame):
                                                train_x, test_x = ori_train_x.iloc[train_index], ori_train_x.iloc[
                                                    test_index]
                                            else:
                                                train_x, test_x = ori_train_x[train_index], ori_train_x[test_index]
                                        else:
                                            train_x, test_x = None, None
                                        with warnings.catch_warnings():
                                            try:
                                                warnings.simplefilter("ignore")
                                                model = sm.tsa.statespace.SARIMAX(train, exog=train_x, order=(p, q, r),
                                                                                  seasonal_order=(
                                                                                  P, Q, R, seasonal_period),
                                                                                  trend=trend)
                                                result = model.fit(disp=False)
                                                prediction = result.predict(start=(len(train)),
                                                                            end=(len(train) + len(test) - 1),
                                                                            exog=test_x)
                                            except np.linalg.linalg.LinAlgError:
                                                # print('error with parameter', [(p,q,r), (P,Q,R,12), trend])
                                                continue
                                        # check mle convergence
                                        if not result.mle_retvals['converged']:
                                            continue

                                        score.append(score_function(test[test.notna()], prediction[test.notna()]))
                                    score = np.nanmean(score)
                                else:
                                    with warnings.catch_warnings():
                                        try:
                                            warnings.simplefilter("ignore")
                                            model = sm.tsa.statespace.SARIMAX(train, exog=train_x, order=(p, q, r),
                                                                              seasonal_order=(P, Q, R, seasonal_period),
                                                                              trend=trend)
                                            result = model.fit(disp=False)
                                            prediction = result.predict(start=(len(train)), end=(len(train) + len(test) - 1), exog=test_x)
                                        except np.linalg.linalg.LinAlgError:
                                            # print('error with parameter', [(p,q,r), (P,Q,R,12), trend])
                                            continue
                                    # check mle convergence
                                    if not result.mle_retvals['converged']:
                                        continue
                                    score = score_function(test, prediction)
                                if score < best_score:
                                    best_params = [(p, q, r), (P, Q, R, seasonal_period), trend]
                                    best_score = score
    return best_params, best_score, best_r2
