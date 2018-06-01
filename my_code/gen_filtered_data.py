from __future__ import division
import pandas as pd
import numpy as np
import networkx as nx
import utilities as ut
from matplotlib import pyplot as plt
from tabulate import tabulate
import os

path = r'C:\Users\Javier\Documents\MEGA\Thesis\filtered_data_original'
filename = 'entities_data.npy'
entities_np = dp.load_entities_data(path, filename)

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
    In fact it not exactly ten days, but the number var_param: that is the delay used
    for calculating the standard deviation.

    Parameters
    -----------
    local_index: index of local minima and local maxima
    entity_ts: 1d numpy array of time series of an entity
    var_param: integer for variance parameter corresponding to epsilon parameter

    Returns
    -------
    List of standard deviations corresponding to local minima and maxima
    """
    stddev_list = []
    for index in local_index:
        if index < var_param:
            stddev_list.append(np.std(entity_ts[0:var_param], dtype=float, ddof=1))
        else:
            stddev_list.append(np.std(entity_ts[index - var_param + 1: index + 1], dtype=float, ddof=1))
    return stddev_list

def compute_epsilon_drawup(entity_ts, epsilon_up_time_parameter, minimal_epsilon_up, absolute_up):
    """
    Calculates epsilon-draw-ups for the time series of an entity based on both standard deviation
    and percentage draw-ups and draw-downs

    Parameters
    ----------
    entity_ts: 1d Numpy array corresponding to the time series of the entity
    epsilon_choice: string parameter corresponding to standard deviation or percentage epsilon
    epsilon_time_parameter: int parameter corresponding to standard deviation calculation    
    Returns
    -------
    List of local minima and maxima indices and epsilon draw-ups
    """

    local_min_index, local_max_index = compute_local_minmax(entity_ts)
    epsilon_up_list_relative = calculate_sd_epsilon(local_min_index, entity_ts, epsilon_up_time_parameter)
    #print "epsilon values are ", epsilon_up_list_relative

    '''if local_min_index[0] < local_max_index[0]:
        epsilon_index_list_relative = calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list_relative, epsilon_down_list_relative,
                                           local_min_index, local_max_index, 0)
    else:
        epsilon_index_list_relative = calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list_relative, epsilon_down_list_relative,
                                           local_min_index, local_max_index, 1)

    #elif epsilon_choice == 'percentage':
    
    if local_min_index[0] < local_max_index[0]:
        epsilon_index_list_percentage = calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list_percentage, epsilon_down_list_percentage,
                                           local_min_index, local_max_index, 0)
    else:
        epsilon_index_list_percentage = calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list_percentage, epsilon_down_list_percentage,
                                           local_min_index, local_max_index, 1)
    '''
    epsilon_index_list_relative = revised_calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list_relative,
                                           local_min_index, local_max_index)
    epsilon_up_list_percentage= \
        calculate_percent_epsilon(local_min_index, local_max_index,
                                  entity_ts, minimal_epsilon_up)
    epsilon_index_list_percentage = revised_calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list_percentage,
                                                                epsilon_down_list_percentage,
                                                                local_min_index, local_max_index)

    epsilon_up_list_absolute = [absolute_up for i in range(len(epsilon_up_list_percentage))]
    epsilon_index_list_absolute = revised_calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list_absolute,
                                                                        local_min_index, local_max_index)
    #print 'relative list', epsilon_index_list_relative
    #print 'percentage list', epsilon_index_list_percentage
    #return list(epsilon_index_list_relative)
    tempList = list(set(epsilon_index_list_percentage).intersection(set(epsilon_index_list_relative)))
    return np.sort(list(set(tempList).intersection(set(epsilon_index_list_absolute))))


def revised_calibrate_epsilon_drawup_ts(entity_ts, epsilon_up_list,\
			local_min_index, local_max_index):
	# Needs to start with a minima
    if local_min_index[0] < local_max_index [0]:
        # Needs to end with a max
        if local_min_index[-1] > local_max_index[-1]:
            local_min_index = local_min_index[0:-1]
            epsilon_up_list = epsilon_up_list[0:-1]
            #local_max_index = local_max_index[0:-2]
            #epsilon_down_list = epsilon_down_list[0:-2]
    else:
        local_max_index = local_max_index[1:]
        # Needs to end with a max
        if local_min_index[-1] > local_max_index[-1]:
            local_min_index = local_min_index[0:-1]
            epsilon_up_list = epsilon_up_list[0:-1]
            #local_max_index = local_max_index[0:-2]
            #epsilon_down_list = epsilon_down_list[0:-2]
    epsilon_drawup_list = []
    i = 0
    index_correction = 0
    if(np.abs(len(local_min_index)-len(local_max_index)))>2:
        index_correction = np.abs(len(local_min_index)-len(local_max_index))

    while i < len(local_min_index)-2 - index_correction:
        j = i
        current_epsilon_drawup_length = len(epsilon_drawup_list)
        if len(epsilon_drawup_list) == current_epsilon_drawup_length and j < len(local_min_index)-2:
            if entity_ts[local_max_index[j]] - entity_ts[local_min_index[i]] > epsilon_up_list[i]:
                epsilon_drawup_list.append(local_min_index[i])
                j += 1
            else:
                j += 1
        i = j
    return epsilon_drawup_list

    def calculate_percent_epsilon(local_min_index, local_max_index, entity_ts, up_percent):
    """
    Calculates the percentage draw-up and draw-down at the local minima
    and local maxima indices respectively

    Parameters
    ----------
    local_min_index: list of local minima
    local_max_index: list of local maxima
    entity_ts: 1d numpy array of time series
    up_percent: float value between 0 and 1 corresponding to percentage draw-up

    Returns
    -------
    epsilon_up_list: list of percentage draw-up for local minima indices
    epsilon_down_list: list of percentage draw-down for local maxima indices
    """

    epsilon_up_list = [up_percent * entity_ts[loc_min] for loc_min in local_min_index]
    return epsilon_up_list

def filter_market_effect(epsilon_drawup_matrix, market_drawup, time_lag):
	



def filter_market_effect(epsilon_drawup_matrix, market_ts, epsilon_choice, time_lag, epsilon_down_time_parameter, epsilon_down_scale, minimal_epsilon_down, absolute_down,
                           epsilon_up_time_parameter, epsilon_up_scale, minimal_epsilon_up, absolute_up):
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
        market_epsilon_drawup_list = compute_epsilon_drawup(market_ts,epsilon_up_time_parameter, minimal_epsilon_up, absolute_up)
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


def order_diff(first, second):
    """
    Calculates the set difference of two input lists
    """
    first = np.ravel(first)
    second = np.ravel(second)
    return [item for item in first if item not in second]
