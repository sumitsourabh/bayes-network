#!C:\Users\Javier\Anaconda3\python

''' Script to generate toy Bayesian network where
	X is the parent of both Y and Z, as the graph shows:

	X ----> Y
	|
	v
	Z

	All the variables are Bernoulli variables.

	These are the following parameters of the network:
	P(X=0) = p_x0
	P(Y=0|X=0) = p_y0_x0
	P(Y=0|X=1) = p_y0_x1
	P(Z=0|X=0) = p_z0_x0
	P(Z=0|X=1) = p_z0_x1
'''


import numpy as np 
import pandas as pd
import os


p_x0 = .3
p_y0_x0 = .9
p_y0_x1 = .4
p_z0_x0 = .5
p_z0_x1 = .2

samples = 100


X = (np.random.rand(samples)>p_x0).astype(int)
Y = []
for x in X:
    if x==0:
        Y.append(np.random.rand()>p_y0_x0)
    else:
        Y.append(np.random.rand()>p_y0_x1)
Y = np.array(Y).astype(int)
Z = []
for x in X:
    if x==0:
        Z.append(np.random.rand()>p_z0_x0)
    else:
        Z.append(np.random.rand()>p_z0_x1)
Z = np.array(Z).astype(int)

data = pd.DataFrame([X,Y,Z], index = ['X','Y','Z'])
data.to_excel(os.path.join(r'C:\Users\Javier\Documents\MEGA\Universitattt\Master\Thesis\CDS_data\toy_data',\
				'toydata.xlsx'))

''' Script to generate toy Bayesian network with a V-structure where
	X and Y are the parents of Z, as the graph shows:

	X     Y
	 \   /
	  \ /
	   V
	   Z

	All the variables are Bernoulli variables.

	These are the following parameters of the network:
	P(X=0) = p_x0
	P(Y=0) = p_y0
	Z = X*Y
	So
	P(Z=1|X,Y)
'''

# V-structure:
p_x0 = .3
p_y0 = .6

X = (np.random.rand(samples)>p_x0).astype(int)
Y = (np.random.rand(samples)>p_y0).astype(int)
Z = X*Y

data = pd.DataFrame([X,Y,Z], index = ['X','Y','Z'])
data.to_excel(os.path.join(r'C:\Users\Javier\Documents\MEGA\Universitattt\Master\Thesis\CDS_data\toy_data',\
				'toy_vstructure.xlsx'))