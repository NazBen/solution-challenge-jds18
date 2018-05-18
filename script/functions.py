import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def normalized_root_mean_squared_error(y_true, y_pred):
    """
    Source: https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / np.mean(y_true) * 100


def check_stationarity(serie):
    result = adfuller(serie)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def get_error_gbm(error_func, target, lba_boxcox=0):
    def gbm_error_func(y_pred, y_true):
        if target == 'log_y':
            y_pred = np.exp(y_pred)
            y_true = np.exp(y_true)
        if target == 'boxcox_y':
            y_pred = inv_boxcox(y_pred, lba_boxcox)
            y_true = inv_boxcox(y_true, lba_boxcox)
        eval_result = error_func(y_pred, y_true)
        return 'error', eval_result, False
    return gbm_error_func
