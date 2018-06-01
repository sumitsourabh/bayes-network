from __future__ import division

import pandas as pd
import numpy as np


def get_entity_data():
    return pd.read_csv('/Users/Sumit/Dropbox/Russianess/data/russian+evraz_data.csv')


def get_date_list():
    date_df = pd.read_csv('/Users/Sumit/Dropbox/Russianess/data/completeDatelist.csv')
    date_list = list(date_df.loc[1260:1498]['MTM_DATE'])
    return date_list


def create_numpy_array(entities_df, date_list):
    [row, col] = entities_df.shape
    entities_np = np.empty([row + 1, (col/3)-1], dtype=object)
    col_list = ['NORM_SPREAD ' + date for date in date_list]
    entities_list = list(entities_df['SHORTNAME']) + ['FX']
    entities_np[:, 0] = entities_list
    for i in range(row):
        entities_np[i, 1:] = 10000 * np.ravel(entities_df[(entities_df['SHORTNAME'] ==
                                                 entities_list[i])][col_list])
    fx_ts = data_preprocess_fx()
    entities_np[row, 1:] = fx_ts
    np.save('/Users/Sumit/Dropbox/Russianess/data/entities_data', entities_np)


def load_entities_data(path, filename):
    return np.load(path+filename)


def load_market_data(path, filename):
    return np.load(path+filename)


def data_preprocess_fx():
    """
    Pre-processes the  pre-processed fx time series data

    Returns
    -------
    Numpy array of USD/RUB fx rate
    """
    fx_df = pd.read_excel('/Users/Sumit/Dropbox/Russianess/data/fxRates.xlsx')
    fx_df['USD RUB FX rate'] = fx_df['RUB EUR FX rate'] / fx_df['USD EUR FX rate']
    return 1. / np.asarray(list(fx_df['USD RUB FX rate'])[346:585])


def data_preprocess_index():
    """
    Returns the list of iTraxx spreads corresponding to the date list

    Parameters
    ----------
    List of dates used for calibration

    Returns
    -------
    Numpy array of iTraxx spreads
    """
    index_df = pd.read_excel('/Users/Sumit/Dropbox/Russianess/data/CDX+iTraxx.xlsx')
    return np.asarray(list(index_df['ITRX EUR CDSI GEN 5Y CBIN CORP']))[1793:2032]


def create_market_numpy():
    """
    Creates and saves the numpy array corresponding to market time series
    """
    market_ts = data_preprocess_index()
    np.save('/Users/Sumit/Dropbox/Russianess/data/market_data', market_ts)



def get_test_data():
    path = '/Users/Sumit/Dropbox/Russianess/data/'
    file_name = 'test_data.xlsx'
    return pd.read_excel(path+file_name)
