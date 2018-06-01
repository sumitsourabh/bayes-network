from __future__ import division

import numpy as np
import networkx as nx
import utilities as ut


def compute_local_minmax(entity_ts):
    """
    Returns the index of local min and
    max for the CDS time series of an entity

    Parameters
    ----------
    entity_ts: 1-D np array of normalised
    CDS time series

    Returns
    -------
    loc_min, loc_max: np array of indices corresponding to local minima and maxima
    """
    local_min_index = (np.diff(np.sign(np.diff(entity_ts))) > 0).nonzero()[0] + 1  # local min
    local_max_index = (np.diff(np.sign(np.diff(entity_ts))) < 0).nonzero()[0] + 1  # local max
    return local_min_index, local_max_index


def calculate_sd_epsilon(local_index, entity_ts, var_param):
    """
    Calculates the standard deviation at the local minima and local maxima of an entity's
    time series of the last ten days from the current date. If the current date is within
    first ten days, it calculates the standard deviation of first 10 days.

    Parameters
    -----------
    local_index: index of local minima and local maxima
    entity_ts: 1d numpy array of time series of an entity
    var_param: integer for variance parameter corresponding to epsilon parameter

    Returns
    -------
    List of standard deviations corresponding to local minima and maxima
    """
    epsilon_list = []
    for index in local_index:
        if index < var_param:
            epsilon_list.append(np.std(entity_ts[0:var_param], dtype=float, ddof=1))
        else:
            epsilon_list.append(np.std(entity_ts[index - var_param + 1: index + 1], dtype=float, ddof=1))
    return epsilon_list


def calculate_percent_epsilon(local_min_index, local_max_index,
                              entity_ts, up_percent, down_percent):
    """
    Calculates the percentage draw-up and draw-down at the local minima
    and local maxima indices respectively

    Parameters
    ----------
    local_min_index: list of local minima
    local_max_index: list of local maxima
    entity_ts: 1d numpy array of time series
    up_percent: float value between 0 and 1 corresponding to percentage draw-up
    down_percent: float value between 0 and 1 corresponding to percentage draw-down

    Returns
    -------
    epsilon_up_list: list of percentage draw-up for local minima indices
    epsilon_down_list: list of percentage draw-down for local maxima indices
    """
    epsilon_up_list = [up_percent * entity_ts[loc_min] for loc_min in local_min_index]
    epsilon_down_list = [down_percent * entity_ts[loc_max] for loc_max in local_max_index]
    return epsilon_up_list, epsilon_down_list


def compute_epsilon_drawup(entity_ts, epsilon_choice):
    """
    Calculates epsilon-draw-ups for the time series of an entity based on both standard deviation
    and percentage draw-ups and draw-downs

    Parameters
    ----------
    entity_ts: 1d Numpy array corresponding to the time series of the entity
    epsilon_choice: string parameter corresponding to standard deviation or percentage epsilon

    Returns
    -------
    List of local minima and maxima indices and epsilon draw-ups
    """
    local_min_index, local_max_index = compute_local_minmax(entity_ts)
    if epsilon_choice == 'std_dev':
        epsilon_up_list = calculate_sd_epsilon(local_min_index, entity_ts, 10)
        epsilon_down_list = calculate_sd_epsilon(local_max_index, entity_ts, 10)
    elif epsilon_choice == 'percentage':
        epsilon_up_list, epsilon_down_list = \
            calculate_percent_epsilon(local_min_index, local_max_index,
                                      entity_ts, 0.04, 0.02)
    if local_min_index[0] < local_max_index[0]:
        return calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list, epsilon_down_list,
                                           local_min_index, local_max_index, 0)
    else:
        return calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list, epsilon_down_list,
                                           local_min_index, local_max_index, 1)


def calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list, epsilon_down_list, local_min_index,
                                local_max_index, order_diff):
    """
    Calibrates the epsilon draw-ups based on input parameters

    Parameters
    ----------
    entity_ts: 1d numpy array of time series of spreads
    epsilon_up_list: list of epsilon draw-ups at local minima
    epsilon_down_list: list of epsilon draw-downs at local maxima
    local_min_index: list of indices of local minima
    local_max_index: list of indices of local maxima
    order_diff: difference parameter for ordering local minima and maxima

    Returns
    -------
    epsilon_drawup_list: 1d numpy array of epsilon draw-ups

    """
    epsilon_drawup_list = []
    i = 0
    j = i + order_diff
    while i < len(local_min_index) + order_diff - 2:
        if (entity_ts[local_max_index[j]] - entity_ts[local_min_index[j - order_diff + 1]]) < \
                epsilon_down_list[j]:
            j += 1
        else:
            if entity_ts[local_max_index[j]] - entity_ts[local_min_index[i]] > epsilon_up_list[i]:
                epsilon_drawup_list.append(local_min_index[i])
                j += 1
            else:
                j += 1
        i = j
    return epsilon_drawup_list


def order_diff(first, second):
    """
    Calculates the set difference of two input lists
    """
    first = np.ravel(first)
    second = np.ravel(second)
    return [item for item in first if item not in second]


def epsilon_drawup_network(entities_np, time_lag, market_ts, epsilon_choice):
    """
    Calculates the adjacency weight matrix based on epsilon drawups whose entries denote
     the conditional probabilities

     Parameters
     ----------
     entities_np: 2d numpy matrix of time series of entities
     time_lag: 1d list of time lag
     market_ts: 1d numpy array of market time series
     epsilon_choice: string parameter corresponding to standard deviation or percentage epsilon

     Returns
     -------
     weight_matrix: Numpy adjacency matrix of edge weights
    """
    epsilon_drawup_matrix = []
    entities_list = entities_np[:, 0]
    for i in range(np.size(entities_list)):
        epsilon_drawup_list = compute_epsilon_drawup(entities_np[i, 1:], epsilon_choice)
        epsilon_drawup_matrix.append(epsilon_drawup_list)
    # filtering market effects using market time series
    epsilon_drawup_matrix = filter_market_effect(epsilon_drawup_matrix, market_ts, epsilon_choice,
                                                 time_lag)
    adj_matrix = calculate_conditional_impact(epsilon_drawup_matrix, time_lag)
    weight_matrix = create_weight_matrix(epsilon_drawup_matrix, adj_matrix, time_lag, np.size(entities_np[0,1:]))
    np.fill_diagonal(weight_matrix, 0)
    return weight_matrix


def filter_market_effect(epsilon_drawup_matrix, market_ts, epsilon_choice, time_lag):
    """
    Filters out the market effect from epsilon draw-ups

    Parameters
    ----------
    epsilon_drawup_matrix: 2d Numpy matrix of epsilon draw-ups of entities
    market_ts: 1d numpy array of time series of market entity
    epsilon_choice:  array, parameter for standard deviation or percentage epsilon
    time_lag: array, parameter for time lags

    Returns
    -------
    epsilon_drawup_matrix_corrected: 2d numpy array of epsilon draw-ups after filtering for market
                                    effects
    """
    # adding epsilon draw-ups for the market time series
    epsilon_drawup_matrix_corrected = []
    local_min_index, local_max_index = compute_local_minmax(market_ts)
    if len(local_min_index) == 0 or len(local_max_index) == 0:
        return epsilon_drawup_matrix
    else:
        market_epsilon_drawup_list = compute_epsilon_drawup(market_ts, epsilon_choice)
        for i in range(len(epsilon_drawup_matrix)):
            entity_ts = np.ravel(epsilon_drawup_matrix[i][:])
            # filtering out the market impact for all time lags
            market_drawup = np.ravel(market_epsilon_drawup_list)
            market_drawup_all = market_drawup
            for time in time_lag:
                market_drawup_tmp = [market_drawup[k] + time for k in range(len(market_drawup))]
                market_drawup_all = np.union1d(market_drawup_all, market_drawup_tmp)
            entity_ts_corrected = order_diff(entity_ts, market_drawup_all)
            epsilon_drawup_matrix_corrected.append(entity_ts_corrected)
        epsilon_drawup_matrix_corrected = np.asarray(epsilon_drawup_matrix_corrected)
        return epsilon_drawup_matrix_corrected


def calculate_conditional_impact(epsilon_drawup_matrix, time_lag):
    """
    Corrects the time series for independence

    Parameters
    ----------
    epsilon_drawup_matrix: 2d matrix of epsilon draw-ups
    time_lag: 1d array of time lag

    Returns
    -------
    adj_matrix: 2d numpy adjacency matrix
    """
    # calculating the  presence of co-occurrence of epsilon draw-ups
    rows = np.asarray(epsilon_drawup_matrix).shape[0]
    adj_matrix = np.zeros([rows, rows])
    for i in range(rows):
        entity_ts = np.ravel(epsilon_drawup_matrix[i][:])
        for j in range(rows):
            entity_ts_to_filter = epsilon_drawup_matrix[j]
            occurrence_matrix = np.zeros([len(time_lag), np.size(entity_ts)])
            for time in time_lag:
                entity_ts_tmp = [entity_ts[k] + time for k in range(len(entity_ts))]
                occurrence_matrix[time][:] = np.in1d(entity_ts_tmp, entity_ts_to_filter)
            occurrence = occurrence_matrix[0][:]
            for index in range(1, occurrence_matrix.shape[0]):
                occurrence = np.logical_or(occurrence, occurrence_matrix[index][:])
            true_count = occurrence.sum()
            adj_matrix[i, j] = true_count / len(entity_ts)
    return adj_matrix


def create_weight_matrix(epsilon_drawup_matrix, adj_matrix, time_lag, len_entity_ts):
    """
    Returns weight matrix based for network construction

    Parameters
    ----------
    epsilon_drawup_matrix: 2d numpy matrix of epsilon draw-ups
    adj_matrix: 2d numpy adjacency matrix
    time_lag: 1d array of time lag
    len_entity_ts: int of length of entity time series

    Returns
    -------
    weight_matrix: 2d numpy weight matrix for network construction

    """
    rows, cols = np.shape(adj_matrix)
    weight_matrix = np.zeros([rows, rows])
    for i in range(rows):
        for j in range(rows):
            if adj_matrix[i, j] - (len(epsilon_drawup_matrix[i][:]) *
                                   len(time_lag) / len_entity_ts) > 0:
                weight_matrix[i, j] = adj_matrix[i, j]
    return weight_matrix


def choose_vertex(visited_list, reachable_list, debt_rank, weight_matrix):
    """
    Calculates the path in the graph with the maximum weight starting from a list of
    visited vertices

    Parameters
    ----------
    visited_list: list of visited nodes
    reachable_list: list of reachable nodes
    debt_rank: array of debt rank
    weight_matrix: 2d numpy weight matrix

    Returns
    -------
    List of the path with maximum weight and value of that weight
    """
    path_weight = []
    # reachable_matrix = pd.DataFrame(index=visited_list, columns=reachable_list)
    reachable_matrix = np.empty((len(visited_list), 3), dtype=object)
    i = 0
    for visited in visited_list:
        for reachable in reachable_list:
            reachable_matrix[i, 0] = visited
            reachable_matrix[i, 1] = reachable
            reachable_matrix[i, 2] = debt_rank[i] * weight_matrix[visited, reachable]
            # reachable_matrix.set_value(visited, reachable, debt_rank[i] *
            #                           weight_matrix[visited, reachable])
            path_weight.append([(visited, reachable), debt_rank[i] *
                                weight_matrix[visited, reachable]])
        i += 1
    path_weight = np.asarray(path_weight)
    weight = np.asarray(path_weight[:, 1])
    max_val = np.max(weight)
    index = list(weight).index(max_val)
    return path_weight[index, 0], max_val


def compute_reachable(visited_list, weight_matrix):
    """
    Calculate the list of reachable nodes which can be reached from the list of visited nodes

     Parameters
     ----------
     visited_list: list of nodes visited
     weight_matrix: numpy matrix with edge weights

    Returns
    -------
    reachable_list: list of nodes reachable from visited nodes
    """
    reachable_list = []
    size = np.shape(weight_matrix)[0]
    for node in visited_list:
        for i in range(size):
            if (weight_matrix[node, i] > 0) & (i not in reachable_list):
                reachable_list.append(i)
    return reachable_list


def calculate_network_stats(entities_list, weight_matrix, compute_degree=False):
    """
    Calculates centrality and degree distribution of the network

    Parameters
    ----------
    entities_list: 1d array of entities
    weight_matrix: 2d numpy array of weights calculated from epsilon draw-ups
    compute_degree: Boolean parameter to compute degree, default value is False

    Returns
    -------
    network_stats_np: Numpy array of network statistics
    """
    num_entities = entities_list.size
    if compute_degree:
        network_stats_np = np.empty([num_entities, 3], dtype=object)
    else:
        network_stats_np = np.empty([num_entities, 2], dtype=object)
    g_network = nx.DiGraph()
    g_network.add_nodes_from(range(num_entities))
    for i in range(num_entities):
        for j in range(num_entities):
            if weight_matrix[i, j] > 0:
                g_network.add_edge(i, j, weight=weight_matrix[i, j])
    eig_cen = nx.eigenvector_centrality_numpy(g_network)
    network_stats_np[:, 0] = entities_list
    network_stats_np[:, 1] = eig_cen.values()
    if compute_degree:
        degree = g_network.degree.values()
        network_stats_np[:, 2] = degree
    network_stats_np = network_stats_np[network_stats_np[:, 1].argsort()[::-1]]
    return network_stats_np


def calibrate_most_central_index(weight_matrix, entities_list):
    """
    Calibrates the most central index which can be stressed for debt rank calculation

    Parameters
    ----------
    weight_matrix: numpy matrix with edge weights
    entities_list: list of entities in the data set

    Returns
    -------
    most_central_index: index of the most central node in the network
    """
    network_stats_np = calculate_network_stats(entities_list, weight_matrix)
    most_central_entity = network_stats_np[0, 0]
    most_central_index = np.where(entities_list == most_central_entity)
    print most_central_index
    return most_central_index


def compute_countryrank(weight_matrix, entities_list, stressed_list, initial_stress,
                        draw_network=False):
    """
    Calculates the debt rank based on the debt rank algorithm

    Parameters
    ----------
    weight_matrix: adjacency numpy matrix of edge weights
    entities_list: numpy array of entities from data set
    stressed_list: list of stressed nodes
    initial_stress: float parameter corresponding to the initial level of stress
    draw_network: Boolean parameter for drawing the network
    Returns
    -------
    debt_rank_df: data frame with debt rank of all entities
    """

    visited_list = []
    if stressed_list[0] == 'self-calibrating':
        most_central_index = np.ravel(calibrate_most_central_index(weight_matrix, entities_list))[0]
        visited_list.append(most_central_index)
    else:
        for node in stressed_list:
            visited_list.append(np.ravel(np.where(entities_list == node))[0])
    country_rank = [initial_stress]
    edge_list = []
    edge_weight = []
    while len(visited_list) < len(entities_list):
        reachable_list = compute_reachable(visited_list, weight_matrix)
        reachable_list = order_diff(reachable_list, visited_list)
        edge, max_val = choose_vertex(visited_list, reachable_list, country_rank, weight_matrix)
        edge_list.append(edge)
        edge_weight.append(max_val)
        visited_list.append(edge[1])
        country_rank.append(max_val)
    if draw_network:
        ut.draw_countryrank_network(entities_list, country_rank, visited_list, edge_list,
                                    edge_weight)

    return create_country_rank_np(visited_list, entities_list, country_rank)


def create_country_rank_np(visited_list, entities_list, country_rank):
    """
    Creates a numpy matrix data frame for debt rank of entities

    Parameters
    ----------
    visited_list: list of visited nodes
    entities_list: list of entities
    country_rank: list of dent rank for entities

    Returns
    -------
    country_rank_df: data frame of debt rank for entities

    """
    num_entities = np.size(entities_list)
    country_rank_np = np.empty((num_entities, 2), dtype=object)
    for i in range(num_entities):
        country_rank_np[i, 0] = entities_list[visited_list[i]]
        country_rank_np[i, 1] = country_rank[i]
    return country_rank_np


def weight(weight_matrix, u, v):
    """
    Returns the weight of the edge from u to v
    Parameters
    ----------
    weight_matrix: 2d numpy weight matrix
    u: node index
    v: node index
    """
    return weight_matrix[u][v]


def adjacent(weight_matrix, u):
    """
    Returns the nodes adjacent to a node
    Parameters
    ----------
    weight_matrix: 2d numpy weight matrix
    u: node index

    Returns
    -------
    list_adjacent: list of adjacent nodes
    """
    list_adjacent = []
    for x in range(len(weight_matrix)):
        if weight_matrix[u][x] < 0 and x != u:
            list_adjacent.insert(0, x)
    return list_adjacent


def extract_min(queue):
    """
    Returns the minimum element of the priority queue
    Parameters
    ----------
    queue: priority queue
    """
    q = queue[0]
    queue.remove(queue[0])
    return q


def decrease_key(queue, key):
    """
    Decreases the key of the nodes in q

    Parameters
    ----------
    queue: priority queue
    key: list of indices whose key needs to be updated
    """
    for i in range(len(queue)):
        for j in range(len(queue)):
            if key[queue[i]] < key[queue[j]]:
                s = queue[i]
                queue[i] = queue[j]
                queue[j] = s


def prim_span_tree(weight_matrix, stressed_node, initial_stress):
    """
    Returns the debt rank of the
    Parameters
    ----------
    weight_matrix: 2d numpy array of weights
    stressed_node: index of the stressed node
    initial_stress: float parameter between 0 and 1 for initial level of stress
    Returns
    -------
    parent_list: list of parent nodes for each node
    debt_rak: list of debt rank of each node
    """
    vertices = [i for i in range(np.shape(weight_matrix)[0])]
    parent_list = [None]*len(vertices)
    country_rank = [0]*len(vertices)
    # initialize and set each value of the array key_list (key) to some large number
    key_list = [999999]*len(vertices)

    # initialize the min queue and fill it with all vertices in V
    Q = [0]*len(vertices)
    for u in range(len(Q)):
        Q[u] = vertices[u]

    # set the key of the root to 0
    key_list[stressed_node] = 0
    country_rank[stressed_node] = initial_stress
    decrease_key(Q, key_list)    # maintain the min queue

    # loop while the min queue is not empty
    while len(Q) > 0:
        u = extract_min(Q)    # pop the first vertex off the min queue
        # loop through the vertices adjacent to u
        adj_list = adjacent(weight_matrix, u)
        for v in adj_list:
            w = weight(weight_matrix, u, v)    # get the weight of the edge uv

            # proceed if v is in Q and the weight of uv is less than v's key
            if Q.count(v) > 0 and w < key_list[v]:
                # set v's parent to u
                parent_list[v] = u
                country_rank[v] = country_rank[u] * w
                # v's key to the weight of uv
                key_list[v] = w
                decrease_key(Q, key_list)    # maintain the min queue
    return parent_list, np.abs(country_rank)


def preprocess_weight_matrix(weight_matrix):
    """
    Preprocess the weight matrix by making its entries negative
    Parameters
    ----------
    weight_matrix: 2d numpy array

    """
    weight_matrix = - weight_matrix
    np.place(weight_matrix, weight_matrix == -0, 9999)
    return weight_matrix


def display_numpy_matrix(numpy_matrix, col_label_1, col_label_2):
    """
     Displays numpy matrix in a readable format

    Parameters
    ----------
    numpy_matrix: 2d numpy matrix to display
    col_label_1: str, column label for the first column
    col_label_2: str, column label for the second column
    """
    print "%s | %s" % (col_label_1, col_label_2)
    print "-----------------------"
    for i in range(np.shape(numpy_matrix)[0]):
        print "%s | %8f" % (numpy_matrix[i, 0], numpy_matrix[i, 1])



