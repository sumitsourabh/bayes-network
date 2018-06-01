import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os
import network_calibration as nc


def get_value_at_index(entity_ts, list_index):
    """
    Returns the value of a time series at specified indices

    Parameters
    ----------
    entity_ts: 1d array of time series
    list_index: list of index to slice time series
    """
    local_val = np.empty(len(entity_ts))
    local_val.fill(np.nan)
    for i in range(len(list_index)):
        local_val[list_index[i]] = entity_ts[list_index[i]]
    return local_val


def get_coefficient_of_variance(market_ts):
    """
    Returns the coefficient of variation of the daily returns of market time series

    Parameters
    ----------
    market_ts: 1d array of market time series
    """
    market_ts_list = market_ts.tolist()
    market_return = [np.abs(market_ts_list[i+1] - market_ts_list[i])/market_ts_list[i] for i in range(len(market_ts_list)-1)]
    print(np.std(market_return))
    print(np.mean(market_return))
    return np.std(market_return)/np.mean(market_return)

def convert_weight_matrix_to_csv(weight_matrix):
    l = np.shape(weight_matrix)[0]
    df =pd.DataFrame(columns=['from','to','weight'])
    froml =[]
    tol=[]
    wl=[]
    for i in range(l):
        for j in range(l):
            froml.append(i)
            tol.append(j)
            wl.append(weight_matrix[i,j])
            print(i,j,weight_matrix[i,j])
    df['from']=froml
    df['to'] = tol
    df['weight'] = wl
    df.to_csv('edges.csv',index=False)


def plot_rus_time_series(entities_np):
    start = datetime.datetime(2014, 5, 1)
    end = datetime.datetime(2015, 3, 31)
    date_axis = pd.bdate_range(start, end)
    for i in range(np.shape(entities_np)[0]):
        plt.plot(date_axis, entities_np[i,1:], linewidth=1, label = entities_np[i,0])
    plt.grid('on')
    plt.title('CDS spreads of Russian entities', fontsize=20)
    plt.ylabel('CDS spread in bps', fontsize=20)
    plt.legend(bbox_to_anchor=(1, 1), loc = 2)
    plt.savefig('russian_cds.pdf', bbox_inches='tight')
    plt.show()

def plot_eur_time_series(entities_np):
    start = datetime.datetime(2014, 7, 1)
    end = datetime.datetime(2015, 12, 31)
    date_axis = pd.bdate_range(start, end)
    print(len(date_axis))
    print(len(entities_np[0,:]))
    for i in range(np.shape(entities_np)[0]):
        #if (entities_np[i,0]!='Hellenic Rep'):
        plt.plot(date_axis, entities_np[i,1:], linewidth=1, label = entities_np[i,0])
    plt.grid('on')
    plt.title('Time series of CDS spreads of German entities', fontsize=20)
    plt.ylabel('CDS spread in bps', fontsize=15)
    plt.legend(bbox_to_anchor=(1, 1), loc = 2, fontsize = 10)
    plt.savefig('german.eps')
    plt.show()







def plot_epsilon_drawup_entity(entities_np, entity_name, epsilon_choice, epsilon_down_time_parameter, epsilon_down_scale, minimal_epsilon_down, absolute_down,
                           epsilon_up_time_parameter, epsilon_up_scale, minimal_epsilon_up, absolute_up):
    """
    Plots the time seres of an entity together with local minima, local maxima and
    epsilon draw-up list

    Parameters
    ----------
    entities_np: 2d numpy matrix corresponding to time series of entities
    entity_name: string, name of the entity whose time series has to be plotted
    epsilon_choice: string parameter corresponding to standard deviation or percentage epsilon
    epsilon_time_parameter: int parameter corresponding to standard deviation calculation    

    """
    entity_index = np.where(entities_np[:, 0] == entity_name)
    entity_ts = np.ravel(entities_np[entity_index, 1:])
    #print entity_ts
    #print np.shape(entity_ts)
    #entity_ts = np.ravel(entities_np)
    print(np.shape(entity_ts))
    local_min_index, local_max_index = nc.compute_local_minmax(entity_ts)
    epsilon_drawup_list = nc.compute_epsilon_drawup(entity_ts, epsilon_choice, epsilon_down_time_parameter, epsilon_down_scale, minimal_epsilon_down, absolute_down,
                           epsilon_up_time_parameter, epsilon_up_scale, minimal_epsilon_up, absolute_up)

    print("epsilon= ", epsilon_drawup_list)
    local_min_val = get_value_at_index(entity_ts, local_min_index)
    local_max_val = get_value_at_index(entity_ts, local_max_index)
    #print local_min_index
    get_average_drawup(entity_ts,local_max_index,epsilon_drawup_list)
    epsilon_drawup_val = get_value_at_index(entity_ts, epsilon_drawup_list)
    #start = datetime.datetime(2014, 5, 1)
    #end = datetime.datetime(2015, 4, 15)

    start = datetime.datetime(2014, 5, 1)
    end = datetime.datetime(2015, 3, 31)
    date_axis = pd.bdate_range(start, end)
    print(len(date_axis))
    print()
    plt.plot(date_axis, entity_ts, linewidth=1)
    plt.plot(date_axis, local_min_val, '*', color='green')
    plt.plot(date_axis, local_max_val, '^', color='red')
    plt.plot(date_axis, epsilon_drawup_val, 'o', color='black')
    plt.grid('on')
    plt.title('Modified epsilon drawup for ' + entity_name, fontsize=20)
    plt.ylabel('CDS spread in bps', fontsize=20)
    #plt.xlabel('Date', fontsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.legend(('CDS spread', 'Local minima', 'Local maxima',
                'Modified epsilon draw-up'), loc=2, prop={'size': 20})
    plt.show()

def get_average_drawup(entity_ts, local_max_index, epsilon_drawup_list):
    max_relevant_list = []
    for i in range(len(epsilon_drawup_list)-1):
        max_temp = local_max_index[(local_max_index > epsilon_drawup_list[i]) & (local_max_index < epsilon_drawup_list[i+1])]
        max_relevant_list.append(np.max(max_temp))
    print(max_relevant_list)
    max_values = get_value_at_index(entity_ts, max_relevant_list)
    max_values_filt =  max_values[~np.isnan(max_values)]
    epsilon_values = get_value_at_index(entity_ts, epsilon_drawup_list)
    epsilon_values_filt = epsilon_values[~np.isnan(epsilon_values)]

    abs_diff = np.abs(max_values_filt - epsilon_values_filt[:-1])

    print(np.mean(abs_diff), np.std(abs_diff))

    print("abs = " ,np.mean(abs_diff) + 2*np.std(abs_diff))

def draw_network(entities_list, weight_matrix):
    g = nx.from_numpy_matrix(weight_matrix)
    eig_cen = nx.eigenvector_centrality_numpy(g)
    network_stats_df = pd.DataFrame(columns=['Entities', 'Centrality'])
    network_stats_df['Entities'] = entities_list
    network_stats_df['Centrality'] = eig_cen.values()
    centrality = eig_cen.values()
    pos = nx.spring_layout(g)
    label_list = dict(enumerate(entities_list))
    nx.draw_networkx(g, labels=label_list, node_color='green', node_size=[v * 2000 for v in centrality],
                     edge_color="gray", pos=pos, font_size=8, alpha=0.6)
    plt.axis("off")
    plt.show()


def draw_network_alternative(weight_matrix, entities_list, region_list, draw):

        print(weight_matrix[0,:])
        import networkx as nx
        #H = nx.Graph()
        H = nx.from_numpy_matrix(weight_matrix)
        G = nx.minimum_spanning_tree(H)
        eig_cen = nx.eigenvector_centrality_numpy(G)
        network_stats_df = pd.DataFrame(columns=['Entities', 'Centrality'])
        network_stats_df['Entities'] = entities_list
        network_stats_df['Centrality'] = eig_cen.values()
        sorted_centrality = network_stats_df.sort_values(by=['Centrality'])
        centrality = eig_cen.values()
        print(sorted_centrality)
        if draw == True:
            pos = nx.spring_layout(G)
            label_list = dict(enumerate(entities_list))
            color_list = []
            for region in region_list:
                if (region == 'E.Eur'):
                    color_list.append("red")
                elif region == 'Europe':
                    color_list.append("green")
                else:
                    color_list.append("green")
            nx.draw_networkx(G, labels=label_list, node_color=color_list, node_size=[v * 3000 for v in centrality],
                             edge_color="gray", pos=pos, font_size=8, alpha=0.6)
            plt.axis("off")
            plt.show()


def draw_countryrank_network(entities_list, debt_rank, visited_list, edge_list, edge_weight):
    size = np.size(entities_list)
    mst_matrix = np.zeros([size, size])
    for i in range(len(edge_list)):
        edge = edge_list[i]
        mst_matrix[edge[0], edge[1]] = edge_weight[i]
    label_list = dict(enumerate(entities_list))
    reds = plt.get_cmap('Reds')
    nodes_list = []
    red_array = []
    for i in range(len(entities_list)):
        index = visited_list.index(i)
        if debt_rank[index] == 1:
            red_array.append(reds(0.99))
        else:
            red_array.append(reds(debt_rank[index]))

        debt_rank_node = debt_rank[index]
        nodes_list.append(entities_list[i] + " (" + str(debt_rank_node)[0:4] + ")")
    h = nx.from_numpy_matrix(mst_matrix)
    pos = nx.circular_layout(h)
    eig_cen = nx.eigenvector_centrality_numpy(h)
    centrality = eig_cen.values()
    nx.draw_networkx(h, labels=label_list, edge_color="black",
                     node_color=red_array, node_cmap=reds,
                     node_size=[v * 2000 for v in centrality], pos=pos,
                     font_size=8, width=50 * edge_weight)
    plt.axis('off')
    plt.show()


def write_ctry_rank_results(ctry_rank, filename):
    cpty_df = pd.read_excel('/Users/Sumit/Documents/cds_systemic_analysis/counterparty_list.xlsx')
    entities_list = list(ctry_rank['Entity'])
    country_list = []
    rating_list = []
    sector_list = []
    for e in entities_list:
        cp = cpty_df[cpty_df['SHORTNAME'] == e]
        country_list.append(list(cp['COUNTRY'])[0])
        rating_list.append(list(cp['AV_RATING'])[0])
        sector_list.append(list(cp['SECTOR'])[0])
    ctry_rank['Country'] = country_list
    ctry_rank['Sector'] = sector_list
    ctry_rank['Rating'] = rating_list
    ctry_rank.to_csv(filename)
