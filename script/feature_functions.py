import pandas as pd
import numpy as np
from loading import get_school_holidays
from holidays import France

import lightgbm as lgb


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encoder(train,
                   test,
                   target,
                   gb_features,
                   prior=None,
                   min_samples_leaf=1,
                   smoothing=1,
                   noise_level=0):
    """    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    """
    target_mean = train[target].mean()
    for feature in gb_features:
        if isinstance(feature, list):
            name = '_'.join(feature) + '_mean'
        else:
            name = feature + '_mean'

        if name in train.columns:
            train.drop(name, axis=1, inplace=True)
        if name in test.columns:
            test.drop(name, axis=1, inplace=True)

        # Compute target mean
        averages = train.groupby(by=feature)[target].agg(["mean", "count"])

        # Compute smoothing
        smoothing_vals = 1 / \
            (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

        # Apply average function to all target data
        if prior is None:
            prior_vals = target_mean
        else:
            prior_vals = train.groupby(by=feature)[prior].mean()
            prior_vals[prior_vals.isnull()] = target_mean

        # The bigger the count the less full_avg is taken into account
        averages[target] = prior_vals * \
            (1 - smoothing_vals) + averages["mean"] * smoothing_vals
        averages.drop(["mean", "count"], axis=1, inplace=True)

        # Apply averages to train and test series
        tmp = averages.reset_index().rename(columns={target: 'average'})

        train = pd.merge(train, tmp, on=feature, how='left').rename(
            columns={'average': name})
        # train[name] = (averages[name]*count - train[target])/(1.0*(count - 1).replace(0, 1))
        train.loc[:, name] = add_noise(
            train.loc[:, name].fillna(target_mean), noise_level)

        test = pd.merge(test, tmp, on=feature, how='left').rename(
            columns={'average': name})
        test.loc[:, name] = add_noise(
            test.loc[:, name].fillna(target_mean), noise_level)

    return train, test


def create_date_features(df, time_features=['dayofweek', 'hour', 'dayofyear'],
                         school_holiday_zones=['A', 'B', 'C'],
                         calendar_file="../input/holidays.xml"):
    """Extract new features relative to the date time and calendar.

    Parameters
    ----------
    df : dataframe
        Data with a `date` column as datetime.
    time_features : list, optional (default=['dayofweek', 'hour', 'dayofyear'])
        The new time feature to extract from the datetime.
    school_holiday_zones : list, optional (default=['A', 'B', 'C'])
        French school holiday zone.
    calendar_file : str, optional (default="../input/holidays.xml")
        The xml file location of the French school holiday calendar.

    Returns
    -------
    dataframe
        The input data with new features.
    """

    # The French holidays
    # True if the date is a national holiday
    fr_holidays = France()
    df['national_holiday'] = df['date'].dt.date.apply(
        lambda x: x in fr_holidays)

    # The try/except is not that good, but anyway...
    for feat in time_features:
        try:
            df[feat] = getattr(df['date'].dt, feat)
        except AttributeError:
            pass

    if 'hourofweek' in time_features:
        df['hourofweek'] = df['date'].dt.hour + (df['date'].dt.dayofweek*24)

    if 'week_type' in time_features:
        df['week_type'] = np.nan
        df.loc[df['date'].dt.dayofweek.isin(range(5)), 'week_type'] = 'weekday'
        df.loc[df['date'].dt.dayofweek == 5, 'week_type'] = 'saturday'
        df.loc[df['date'].dt.dayofweek == 6, 'week_type'] = 'sunday'

    # Get school holidays
    start_date_data = df['date'].values[0]
    end_date_data = df['date'].values[-1]
    school_holidays = get_school_holidays(
        start_date_data, end_date_data, zones=school_holiday_zones,
        calendar_file=calendar_file)

    # New feature, True if the date is a school holiday
    for zone in school_holidays:
        feat_name = 'holiday_zone_%s' % (zone)
        df[feat_name] = False
        for period in school_holidays[zone]:
            window = (df['date'] >= period[0]) & (df['date'] <= period[1])
            df.loc[window, feat_name] = True

    return df


def correct_nan_weather(df, quad_features=[], linear_features=[], categorical_features=[],
                        start_blank='2016-02-20 21:00:00', end_blank='2016-03-01 00:00:00'):
    """Correct the missing values from weather data.

    Parameters
    ----------
    df : dataframe
        The weather data with missing values
    quad_features : list, optional (default=[])
        The weather feature for quadratic interpolation.
    linear_features : list, optional (default=[])
        The weather feature for linear interpolation.
    categorical_features : list, optional (default=[])
        The weather feature for nearest interpolation.

    Returns
    -------
    dataframe
        The corrected weather data.
    """

    # There is a blank period in the date in which no weather data have been registered
    start_blank = pd.to_datetime(start_blank)
    end_blank = pd.to_datetime(end_blank)
    blanc_period = (df['date'] > start_blank) & (df['date'] < end_blank)

    df.loc[~blanc_period, quad_features] = df[quad_features].interpolate(
        method='quadratic')
    df.loc[~blanc_period, linear_features] = df[linear_features].interpolate(
        method='linear')
    df.loc[~blanc_period, categorical_features] = df[categorical_features].interpolate(
        method='nearest')
    return df


def correct_history_target(test, targets, n_weeks, n_days_test=8):
    """Clear the history
    """
    for n_week in n_weeks:
        diff = n_days_test * 24 - n_week*7*24
        for feat in targets:
            if diff > 0:
                test.loc[-diff:, '%s_shift_week_%d' % (feat, n_week)] = np.nan

    return test
