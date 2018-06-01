import numpy as np
import pandas as pd
import os
import datetime
import data_proc as dp
import epsilon_module as em
import utilities as ut
from matplotlib import pyplot as plt

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Russian_processed\data'
    filename = r'entities_data.xlsx'
    cds_ts = dp.read_data(path,filename)
    [row, col] = cds_ts.shape
    entities_names = cds_ts.loc[:,0]

    russian_cds = cds_ts.iloc[:-3,:]
    start = datetime.datetime(2014, 5, 1)
    end = datetime.datetime(2015, 3, 31)
    date_axis = pd.bdate_range(start, end)

    russian_names = entities_names[:-3]

    input_market_ts = np.zeros([(russian_cds.shape)[0]-1])

    print('Done loading data:' + str(datetime.datetime.now() - start_time))
    print ('----------------------------------------------------------------')

    # Compute log returns of the cds data
    log_cds = em.compute_log_returns_cds(cds_ts)
    ut.write_to_csv(log_cds, 'log_returns_cds.csv')
    k2, p = em.normal_test(log_cds)

    local_min, local_max = em.compute_local_extremes(cds_ts.iloc[0,1:])
    epsilon_drawups = em.compute_epsilon_drawup(cds_ts.iloc[0,1:],10)


    # ut.plot_eps_drwups_entity(cds_ts.iloc[0,1:], date_axis)

    # ut.plot_cds_all(cds_ts.iloc[:-2,:], date_axis, True)

    table_eps = em.compute_table_epsilon(cds_ts.iloc[:-1,:], date_axis, 10)

    ut.write_to_excel(table_eps, 'raw_epsilon_drawups.xlsx')
    ut.write_to_csv(table_eps, 'raw_epsilon_drawups.csv')

    delay_eps = em.compute_delayed_ep_drawups(table_eps, 3)

    ut.write_to_excel(delay_eps, 'delay_epsilon_drawups.xlsx')
    ut.write_to_csv(delay_eps, 'delay_epsilon_drawups.csv')