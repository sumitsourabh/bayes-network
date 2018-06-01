from __future__ import division

import datetime
import network_calibration as nc
import data_preprocess as dp
import utilities as ut
import pandas as pd
import numpy as np

start_time = datetime.datetime.now()
path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\sumit_data'
filename = 'entities_data.npy'
input_entities_np = dp.load_entities_data(path, filename)
input_entities_list = input_entities_np[:, 0]
input_epsilon_choice = 'std_dev'  # choices are 'std_dev' or 'percentage'
market_filename = 'market_data.npy'
input_market_ts = dp.load_market_data(path, market_filename)
print(['Done loading data:', str(datetime.datetime.now() - start_time)])
print('----------------------------------------------------------------')

# plotting time series with epsilon_draw-ups
# entity_name = 'Russian Agric Bk'
# ut.plot_epsilon_drawup_entity(input_entities_np, entity_name)

input_time_lag = [0, 1, 2]
weight_matrix = nc.epsilon_drawup_network(input_entities_np, input_time_lag, input_market_ts,
                                          input_epsilon_choice)

network_stats_np = nc.calculate_network_stats(input_entities_np[:, 0], weight_matrix)
print(pd.DataFrame(data=network_stats_np, columns=['Entities', 'Centrality']))

# plotting network based on weight matrix
# ut.draw_network(input_entities_np[:, 0], weight_matrix)
print('----------------------------------------------------------------')
print(['Done calibrating network weight matrix:', str(datetime.datetime.now() - start_time)])
print('----------------------------------------------------------------')

# choices are self-calibrating or name of the entity to stress
input_stressed_list = ['self-calibrating']
# float parameter between 0 and 1 corresponding to initial level of stress for stressed entity
input_initial_stress = 1


country_rank_np = nc.compute_countryrank(weight_matrix, input_entities_list, input_stressed_list,
                                         input_initial_stress, draw_network=False)
temp = pd.DataFrame(data=country_rank_np, columns=['Entities', 'CountryRank'])
temp_net = pd.DataFrame(data=network_stats_np, columns=['Entities', 'Centrality'])
temp.to_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\sumit_data\countryrank.csv')
temp_net.to_csv(r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\sumit_data\net_stats.csv')
print(pd.DataFrame(data=country_rank_np, columns=['Entities', 'CountryRank']))
print('----------------------------------------------------------------')
print(['Done evaluating CountryRank:', str(datetime.datetime.now() - start_time)])
print('----------------------------------------------------------------')


