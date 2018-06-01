import unittest
import numpy as np
from numpy import linalg as la
from network_wwr import network_calibration as nc
from network_wwr import data_preprocess as dp


class NetworkTest(unittest.TestCase):

    def test_compute_local_minmax(self):
        """
        Test for the compute_local_minmax function
        """
        def f(x): return x ** 4 - 2 * x ** 2 + 3
        f_ts = np.asarray([f(i) for i in range(-5, 5, 1)])
        self.assertTrue((nc.compute_local_minmax(f_ts)[0] == np.asarray([4, 6])).all())
        self.assertTrue((nc.compute_local_minmax(f_ts)[1] == np.asarray([5])).all())
        test_pd = dp.get_test_data()
        test_ts = np.asarray(list(test_pd['ts']))
        test_ts_min_index = np.asarray(list(test_pd['local_min_index']))
        test_ts_max_index = np.asarray(list(test_pd['local_max_index']))
        test_ts_min_index = test_ts_min_index[~np.isnan(test_ts_min_index)]
        test_ts_max_index = test_ts_max_index[~np.isnan(test_ts_max_index)]
        self.assertTrue(np.array_equal(nc.compute_local_minmax(test_ts)[0], test_ts_min_index))
        self.assertTrue(np.array_equal(nc.compute_local_minmax(test_ts)[1], test_ts_max_index))

    def test_calculate_sd_epsilon(self):
        """
        Test for the calculate_sd_epsilon function
        """
        test_pd = dp.get_test_data()
        test_ts = np.asarray(list(test_pd['ts']))
        test_sd = list(test_pd['sd'])
        ts_min_index, ts_max_index = nc.compute_local_minmax(test_ts)
        test_sd_min = [test_sd[index] for index in ts_min_index]
        print nc.calculate_sd_epsilon(ts_min_index, test_ts, 10)
        print np.asarray(test_sd_min)
        self.assertTrue(la.norm(nc.calculate_sd_epsilon(ts_min_index, test_ts, 10) - np.asarray(test_sd_min)) < 0.0001)

    def test_compute_epsilon_drawup(self):
        """
        Test for the compute_epsilon_drawup function
        """
        test_pd = dp.get_test_data()
        test_ts = np.asarray(list(test_pd['ts']))
        test_ts_epsilon_min = np.asarray(list(test_pd['epsilon_min_index']))
        test_ts_epsilon_min = test_ts_epsilon_min[~np.isnan(test_ts_epsilon_min)]
        ts_epsilon_min = nc.compute_epsilon_drawup(test_ts, 'std_dev')
        self.assertTrue(np.array_equal(ts_epsilon_min, test_ts_epsilon_min))

    def test_epsilon_drawup_network(self):
        test_ts = np.random.rand(50)
        test_entities_np = np.empty((3, 1 + np.size(test_ts)), dtype=object)
        test_entities_np[:, 0] = ['a', 'b', 'c']
        for i in range(3):
            test_entities_np[i, 1:] = test_ts
        test_time_lag = [0]
        test_market_ts = np.zeros(np.size(test_ts))
        test_epsilon_choice = 'std_dev'
        weight_matrix = nc.epsilon_drawup_network(test_entities_np, test_time_lag, test_market_ts,
                                                  test_epsilon_choice)
        test_weight_matrix = np.ones([3, 3])
        np.fill_diagonal(test_weight_matrix, 0)
        self.assertTrue(np.array_equal(test_weight_matrix, weight_matrix))

    def test_filter_market_effect(self):
        test_ts = np.random.rand(50)
        test_entities_np = np.empty((3, 1 + np.size(test_ts)), dtype=object)
        test_entities_np[:, 0] = ['a', 'b', 'c']
        for i in range(3):
            test_entities_np[i, 1:] = test_ts
        test_time_lag = [0]
        test_market_ts = test_ts
        test_epsilon_choice = 'std_dev'
        weight_matrix = nc.epsilon_drawup_network(test_entities_np, test_time_lag, test_market_ts,
                                                  test_epsilon_choice)
        test_weight_matrix = np.zeros([3, 3])
        print weight_matrix
        self.assertTrue(np.array_equal(test_weight_matrix, weight_matrix))

    def test_compute_countryrank(self):
        """
        Test for the compute_country_rank function
        """
        test_weight_matrix = np.zeros([5, 5])
        test_weight_matrix[0, 1] = 0.9
        test_weight_matrix[0, 2] = 0.8
        test_weight_matrix[0, 3] = 0.2
        test_weight_matrix[1, 4] = 0.3
        test_weight_matrix[2, 3] = 0.6
        test_entities_list = np.array(['a', 'b', 'c', 'd', 'e'])
        test_stressed_list = ['a']
        test_initial_stress = 1
        test_country_rank = np.array([1, 0.9, 0.8, 0.48, 0.27])
        country_rank = nc.compute_countryrank(test_weight_matrix, test_entities_list, test_stressed_list,
                                              test_initial_stress)
        print country_rank
        debt_rank_array = np.asarray(country_rank[:, 1])
        self.assertTrue(np.array_equal(debt_rank_array, test_country_rank))


if __name__ == '__main__':
    unittest.main()
