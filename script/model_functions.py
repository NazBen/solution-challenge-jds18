import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import inv_boxcox
from statsmodels.tsa.arima_model import ARIMAResults, ARMAResults

from functions import (check_stationarity, mean_absolute_percentage_error,
                       normalized_root_mean_squared_error)


def build_gbm_model(X_train, y_train, X_test, y_test, gbm_params, early_stopping_rounds=500, error='mape', verbose=500):
    gbm = lgb.LGBMRegressor(**gbm_params)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=error,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds
            )
    return gbm


def get_df_arma(train, target):
    df_target = train.loc[~train[target].isnull(), ['date', target]]
    df_target.index = df_target['date']
    del df_target['date']
    return df_target


def build_arma_model(train, target, params=(24, 24), start_params=None, adfuller_test=True, start_at_lags=0, maxiter=100):

    df_target = get_df_arma(train, target)

    if adfuller_test:
        check_stationarity(df_target[target])

    model = sm.tsa.ARMA(df_target, params)
    start = train['date'].head(1)[0].strftime('%d_%m_%Y_%H_%M')
    end = pd.to_datetime(train['date'].tail(
        1).values[0]).strftime('%d_%m_%Y_%H_%M')

    name = '../models/arma_model'
    if len(train) == 8767:
        name += '_full'
    elif len(train) == 8768:
        name += '_full_with_last_test'
    filename = name + '_p_%d_q_%d_%s_from_%s_to_%s.pkl' % (
        *params, target, start, end)
    if os.path.exists(filename):
        result = ARMAResults.load(filename)
    else:
        result = model.fit(start_params=start_params,
                           solver='bfgs', start_at_lags=start_at_lags)
        result.save(filename)

    return result


def build_arima_model(train, target, params=(24, 0, 24), adfuller_test=True, maxiter=100):

    df_target = get_df_arma(train, target)

    if adfuller_test:
        check_stationarity(df_target[target])

    model = sm.tsa.ARIMA(df_target, params)
    start = train['date'].head(1)[0].strftime('%d_%m_%Y_%H_%M')
    end = pd.to_datetime(train['date'].tail(
        1).values[0]).strftime('%d_%m_%Y_%H_%M')

    filename = '../models/arima_model_p_%d_d_%d_q_%d_%s_from_%s_to_%s.pkl' % (
        *params, target, start, end)
    if os.path.exists(filename):
        print("Loading...")
        result = ARIMAResults.load(filename)
    else:
        result = model.fit(solver='bfgs', maxiter=maxiter)
        result.save(filename)

    return result


def pred_arma_test(model, train, test, target, lba_boxcox=None):
    if 'log_y' in target:
        feat = 'log_y'
    elif 'boxcox_y' in target:
        feat = 'boxcox_y'
    else:
        feat = 'y'
    n = test.shape[0]
    forecast = model.forecast(n)[0]
    if target == 'custom_y':
        print(forecast.shape)
    else:
        if 'diff' in target:
            if '_h' in target:
                y0 = train[feat].values[-1]
                y = y0 + forecast.cumsum()
        else:
            y = forecast

        if 'log_y' in target:
            y = np.exp(y)
        elif 'boxcox_y' in target:
            y = inv_boxcox(y, lba_boxcox)
    return y


def pred_arma_train(model, train, target, lba_boxcox=None):
    if 'log_y' in target:
        feat = 'log_y'
    elif 'boxcox_y' in target:
        feat = 'boxcox_y'
    else:
        feat = 'y'

    arma_predict = model.predict().values
    if target == 'custom_y':
        print(arma_predict.shape)
    else:
        if 'diff' in target:
            if '_h' in target:
                y = np.r_[train[feat].values[0], arma_predict +
                          train[feat].shift().values[1:]]
        else:
            y = arma_predict

        if 'log_y' in target:
            y = np.exp(y)
        elif 'boxcox_y' in target:
            y = inv_boxcox(y, lba_boxcox)
    return y


def create_local_train_test(df, start_train, end_train, n_days_test=8):
    """Creates local train/test split.

    Parameters
    ----------
    df : dataframe
        Complete data.
    start_train : str
        Starting date for train.
    end_train : str
        Ending date for train.
    n_days_test : int, optional (default=8)
        Number of days in the local test.

    Returns
    -------
    train : dataframe
        The local train.
    test : dataframe
        The local test.
    """

    start_train = pd.to_datetime(start_train, dayfirst=True)
    end_train = pd.to_datetime(end_train, dayfirst=True)

    # The rows in the local train
    mask_train = (df['date'] >= start_train) & (df['date'] < end_train)

    # End test
    start_test = end_train
    end_test = end_train + pd.DateOffset(n_days_test)

    # The rows in the local test
    mask_test = (df['date'] >= start_test) & (df['date'] <= end_test)

    # Local train and test
    train = df[mask_train].copy()
    test = df[mask_test].copy()

    return train, test
