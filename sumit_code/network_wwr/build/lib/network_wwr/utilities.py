import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import datetime
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


def plot_epsilon_drawup_entity(entities_np, entity_name):
    """
    Plots the time seres of an entity together with local minima, local maxima and
    epsilon draw-up list

    Parameters
    ----------
    entities_np: 2d numpy matrix corresponding to time series of entities
    entity_name: string, name of the entity whose time series has to be plotted
    """
    entity_index = np.where(entities_np[:, 0] == entity_name)
    entity_ts = np.ravel(entities_np[entity_index, 1:])
    local_min_index, local_max_index = nc.compute_local_minmax(entity_ts)
    epsilon_drawup_list = nc.compute_epsilon_drawup(entity_ts, 'std_dev', 'min')
    local_min_val = get_value_at_index(entity_ts, local_min_index)
    local_max_val = get_value_at_index(entity_ts, local_max_index)
    epsilon_drawup_val = get_value_at_index(entity_ts, epsilon_drawup_list)
    start = datetime.datetime(2014, 5, 1)
    end = datetime.datetime(2015, 3, 31)

    date_axis = pd.bdate_range(start, end)
    plt.plot(date_axis, entity_ts, linewidth=2)
    plt.plot(date_axis, local_min_val, 'o', color='green')
    plt.plot(date_axis, local_max_val, 'o', color='red')
    plt.plot(date_axis, epsilon_drawup_val, 'o', color='black')
    plt.grid('on')
    plt.title('Modified epsilon draw-up for ' + entity_name, fontsize=20)
    plt.ylabel('CDS spread in bps', fontsize=15)
    plt.legend(('CDS spread', 'Local minima', 'Local maxima',
                'modified epsilon draw-up'), loc=2)
    plt.show()


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
    pos = nx.spring_layout(h)
    eig_cen = nx.eigenvector_centrality_numpy(h)
    centrality = eig_cen.values()
    nx.draw_networkx(h, labels=label_list, edge_color="black",
                     node_color=red_array, node_cmap=reds,
                     node_size=[v * 2000 for v in centrality], pos=pos,
                     font_size=8, width=50 * edge_weight)
    plt.axis('off')
    plt.show()
