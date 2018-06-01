import pandas as pd
import numpy as np 
import datetime as datetime
import sys 
import os
sys.path.append(os.path.abspath(r'C:\Users\Javier\Documents\MEGA\Universitattt\Master\Thesis\Code'))
from FunctionsCDS import *

date_df = pd.read_csv('/Users/Sumit/Dropbox/Russianess/completeDatelist.csv')
date_list = list(date_df.loc[1303:1695]['MTM_DATE'])
print(date_list[0], date_list[-1])
#counterparty_data = select_counterparties_by_region_ctry(region,True)
#print(counterparty_data)
#counterparty_merged_data = merge_counterparty_data(counterparty_data,date_list,region,sector,rating)
#print(rsa_data.merge_counterparty_data(counterparty_data,date_list,region,sector,rating))

counterparty_data.groupby(['COUNTRY'])['SHORTNAME'].count()

counterparty_data = select_counterparties_by_ctry('Germany')
#print(counterparty_data)
counterparty_merged_data = merge_counterparty_data(counterparty_data,date_list)
#print(rsa_data.merge_counterparty_data(counterparty_data,date_list,region,sector,rating))

print(counterparty_merged_data.SHORTNAME.unique())

create_numpy_array(counterparty_merged_data, date_list)

date_list_single = list(date_df.loc[1303:1303]['MTM_DATE'])
counterparty_data_all = select_counterparties_by_region_single('E.Eur', date_list_single)

counterparty_data_all[counterparty_data_all['SECTOR'] == 'Government']

counterparty_data_all[counterparty_data_all['SECTOR'] == 'Government']