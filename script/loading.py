import xml.etree.cElementTree as et

import numpy as np
import pandas as pd


def load_data(conso_train_file='../input/conso_train.csv',
              meteo_train_file='../input/meteo_train.csv',
              meteo_test_file='../input/meteo_prev.csv'):
    """"Load the complete dataset from consumption and weather datasets.

    Returns
    -------
    data : dataframe
        The complete data set.
    """
    # Loading the weather data
    train_meteo = pd.read_csv(meteo_train_file, sep=';')
    test_meteo = pd.read_csv(meteo_test_file, sep=';')

    # Marking the type and merge the train/test dataframes
    train_meteo['type'] = 'train'
    test_meteo['type'] = 'test'
    data_meteo = pd.concat([train_meteo, test_meteo])
    # Correct the weather data
    data_meteo = correct_weather_data(data_meteo)

    # Load the train consumption
    train_conso = pd.read_csv(conso_train_file, sep=';')
    # Correct the consumption data
    train_conso = correct_conso_data(train_conso)

    # Mark the type and merge the consumption/weather dataframes
    train_conso['type'] = 'train'
    data = pd.merge(train_conso, data_meteo,
                    on=['date', 'type'], how='outer', sort=True)

    # Complete the test dates
    data = add_test_dates(data)
    # Correct the test data from the merging
    data = correct_test_data(data)

    return data


def correct_conso_data(df):
    """Correct the consumption dates.

    Parameters
    ----------
    df : dataframe
        The consumption data.

    Returns
    -------
    dataframe
        The corrected consumption data.
    """

    date = df['date'].str.split('+', expand=True)
    df['date'] = pd.to_datetime(date[0]).dt.round('1h')
    df['heure_ete'] = date[1].str[:2].astype(float) == 2.
    df.rename(columns={'puissance': 'y'}, inplace=True)
    return df


def correct_weather_data(df):
    """Correct the dates and rename the colums with special characters to much simpler names.

    Parameters
    ----------
    df : dataframe
        The weather data with complicated column names.

    Returns
    -------
    dataframe
        The weather data with new column names.
    """

    columns = {'Date UTC': 'date',
               'T¬∞ (C)': 'temperature',
               'P (hPa)': 'pression',
               'HR (%)': 'HR',
               'P.ros√©e (¬∞C)': 'rosee',
               'Visi (km)': 'visibilite',
               'Vt. moy. (km/h)': 'v_moy',
               'Vt. raf. (km/h)': 'v_raf',
               'Vt. dir (¬∞)': 'v_dir',
               'RR 3h (mm)': 'RR3h',
               'Neige (cm)': 'neige',
               'Nebul. (octats)': 'nebul'}

    df = df.rename(columns=columns)
    df['date'] = df['date'].str.replace('h', ':')
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    return df


def add_test_dates(df, start='2016-09-13 00:00:00', end='2016-09-20 23:00:00'):
    """Adds missing dates in a dataframe.

    Since the dates from the weather data are each three hours and the consumption
    needs to be done every hours, we need to add "fake" rows for each hour.

    Parameters
    ----------
    df : dataframe
        Data with the `type` column that specify the test rows.

    start : str
        The first datetime of the test data.

    end : str
        The last datetime of the test data.

    Returns
    -------
    dataframe
        The completed dataset.
    """
    h_range = pd.date_range(start, end, freq='1H')
    test_dates = pd.DataFrame({'date': h_range})
    test_dates['type'] = 'test'
    df = df.merge(test_dates, on=['date', 'type'], how='outer', sort=True)
    return df


def correct_test_data(df):
    """Correct the test data.

    A slight error in the train/test split forgot to remove the last row of the
    train set, which is also the first row to predict in the test set. Thus,
    we remove the first row in the test and consider it in the train.

    Parameters
    ----------
    df : dataframe
        The train/test data.
    Returns
    -------
    dataframe
        The corrected train/test data.
    """
    # The row that is in bot the train and test
    id_start = df['date'] == "2016-09-13 00:00:00"
    # We copy its information
    df.loc[id_start, :] = df[id_start].fillna(method='bfill')
    # We remove it then and reset the index
    df.drop(np.where(id_start)[0][-1], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Fill the summer hour info
    df.loc[:, 'heure_ete'] = df[['heure_ete']].fillna(method='bfill')
    df.loc[:, 'heure_ete'] = df[['heure_ete']].fillna(method='ffill')
    return df


def get_school_holidays(start_date_data, end_date_data,
                        zones=['A', 'B', 'C'],
                        calendar_file="../input/holidays.xml"):
    """Gets the school holidays between two given dates for specific
    French holiday zones.

    Parameters
    ----------
    start_date_data : str
        Starting date.
    end_date_data : str
        Edning date.
    zones : list, optional (default=['A', 'B', 'C'])
        French school holiday zone.
    calendar_file : str, optional (default="../input/holidays.xml")
        The xml file for school holidays.

    Returns
    -------
    dict
        The school holidays between the starting and ending dates.
    """

    # Not necessary to explain. It basicaly gets the information
    # from the xml file.
    parsedXML = et.parse(calendar_file)
    school_holidays = {}
    node_cal = parsedXML.getroot()[-1]
    for node_zone in node_cal.getchildren():
        zone = node_zone.attrib['libelle']
        if zone in zones:
            dates = []
            for node_date in node_zone.getchildren():
                start = pd.to_datetime(node_date.attrib['debut'])
                end = pd.to_datetime(node_date.attrib['fin'])
                if (end >= start_date_data) and (start <= end_date_data):
                    dates.append([start, end])
            school_holidays[zone] = dates
    return school_holidays
