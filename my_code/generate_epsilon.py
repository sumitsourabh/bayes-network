import numpy as np
import pandas as pd
import os
import datetime
import data_proc as dp
import epsilon_module as em
import utilities as ut

def read_data(path,filename):
	''' This functions reads the file from the path'''
	if filename.endswith('.xlsx'):
		return pd.read_excel(os.path.join(path,filename))
	else:
		return pd.DataFrame(np.load(os.path.join(path,filename)))

def epsilon_first():
	''' This function computes the epsilon draw-ups for
		different time parameters and different delays.'''
	start_time = datetime.datetime.now()
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\Russian_processed\data'
    filename = r'entities_data.xlsx'
    cds_ts = read_data(path,filename)
    [row, col] = cds_ts.shape
    entities_names = cds_ts.loc[:,0]

    russian_cds = cds_ts.iloc[:-3,:]
    start = datetime.datetime(2014, 5, 1)
    end = datetime.datetime(2015, 3, 31)
    date_axis = pd.bdate_range(start, end)

    print('Done loading data:' + str(datetime.datetime.now() - start_time))
    print ('----------------------------------------------------------------')


    ''' 10 days as time parameter'''
    time_para = 10
    epsi_10_raw = em.compute_table_epsilon(cds_ts, date_axis, time_para)
    epsi_10_delay = {}
    for k in range(1,4):
        epsi_10_delay[k] = em.compute_delayed_ep_drawups(epsi_10_raw, k)

    ''' 15 days as time parameter'''
    time_para = 15
    epsi_15_raw = em.compute_table_epsilon(cds_ts, date_axis, time_para)
    epsi_15_delay = {}
    for k in range(1,4):
        epsi_15_delay[k] = em.compute_delayed_ep_drawups(epsi_15_raw, k)


    ''' 20 days as time parameter'''
    time_para = 20
    epsi_20_raw = em.compute_table_epsilon(cds_ts, date_axis, time_para)
    epsi_20_delay = {}
    for k in range(1,4):
        epsi_20_delay[k] = em.compute_delayed_ep_drawups(epsi_20_raw, k)

    ut.epsilon_to_csv(epsi_10_raw, 'raw_ep_drawups_10.csv', 'stddev_10')
    ut.epsilon_to_csv(epsi_15_raw, 'raw_ep_drawups_15.csv', 'stddev_15')
    ut.epsilon_to_csv(epsi_20_raw, 'raw_ep_drawups_20.csv', 'stddev_20')

    for k, v in epsi_10_delay.items():
            ut.epsilon_to_csv(v, 'ep_drawups_10'+'_delay_'+str(k)+'.csv', 'stddev_10')

    for k, v in epsi_15_delay.items():
            ut.epsilon_to_csv(v, 'ep_drawups_15'+'_delay_'+str(k)+'.csv', 'stddev_15')

    for k, v in epsi_20_delay.items():
            ut.epsilon_to_csv(v, 'ep_drawups_20'+'_delay_'+str(k)+'.csv', 'stddev_20')

if __name__ == "__main__":
    