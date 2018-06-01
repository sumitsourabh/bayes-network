#!C:\Users\Javier\Anaconda3\python

'''
    Run with python -mrun name_of_script.py
    Otherwise the whole thing explodes when printing the
    CPDs of the network because they contain unicode characters
'''

import os
import numpy as np 
import pandas as pd 

from pgmpy.estimators import HillClimbSearch, BicScore,\
                        MaximumLikelihoodEstimator, BayesianEstimator


path = r'C:\Users\Javier\Documents\MEGA\Universitattt\Master\Thesis\CDS_data\toy_data'
file1 = r'toydata.xlsx'

data1 = pd.read_excel(os.path.join(path,file1)).transpose()

file2 = r'toy_vstructure.xlsx'
data2 = pd.read_excel(os.path.join(path,file2)).transpose()

print('Using BicScore Method')

for k,data in {file1:data1, file2:data2}.items():
    print('Using ' + k)

    if k==file1:
    	print('True network:')
    	print('Y <-- X --> Z')
    else:
    	print('True network:')
    	print('X --> Z <-- Y')


    est = HillClimbSearch(data, scoring_method = BicScore(data))
    best_model = est.estimate()

    MLE_estimator = MaximumLikelihoodEstimator(best_model, data)
    MLE_parameters = MLE_estimator.get_parameters()

    bay_estimator = BayesianEstimator(best_model, data)
    bay_parameters = bay_estimator.get_parameters()

    print('Learnt edges:')
    print(best_model.edges())

    # print('MLE Parameters')
    # for m in MLE_parameters:
    #     print(m)

    print('Bayesian Parameters')
    for b in bay_parameters:
        print(b)