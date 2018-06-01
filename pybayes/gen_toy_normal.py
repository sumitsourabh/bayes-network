#!C:\Users\Javier\Anaconda3\python

''' Script to generate toy Bayesian network where
	X is the parent of both Y and Z, as the graph shows:

	X ----> Y
	|
	v
	Z

	All the variables are Normal variables.

	These are the following parameters of the network:
	X is standard normal N(mu_x,sigma_x)
	P(Y|x) = N(y_0 + y_1 * x, sigma_y)
	P(Z|x) = N(z_0 + z_1 * x, sigma_z)
'''


import numpy as np 
import pandas as pd
import os


mu_x = 0
sigma_x = 1
# Data for Y
y_0 = 1
y_1 = 2
sigma_y = 1
# Data for Z
z_0 = -1
z_1 = 1
sigma_z = 3

samples = 1000


X = np.random.normal(mu_x, sigma_x, samples)
Y = np.random.normal(y_0 + y_1 * X, sigma_y)
Z = np.random.normal(z_0 + z_1 * X, sigma_z)

data = pd.DataFrame([X,Y,Z], index = ['X','Y','Z'])
data.to_excel(os.path.join(r'C:\Users\Javier\Documents\MEGA\Universitattt\Master\Thesis\CDS_data\toy_data',\
				'toy_normal.xlsx'))

''' Script to generate toy Bayesian network with a V-structure where
	X and Y are the parents of Z, as the graph shows:

	X     Y
	 \   /
	  \ /
	   V
	   Z

	All the variables are Normal variables.

	These are the following parameters of the network:
	X is normal N(mu_x,sigma_x)
	Y is normal N(mu_y,sigma_y)
	P(Z|x,y) = N(z_0 + z_1 * x + z_2 * y, sigma_z)
'''

# V-structure:
mu_x = 1
sigma_x = 1
# Data for Y
mu_y = 0
sigma_y = 3
# Data for Z
z_0 = -1
z_1 = 1
z_2 = -1
sigma_z = 1

samples = 100


X = np.random.normal(mu_x, sigma_x, samples)
Y = np.random.normal(mu_y, sigma_y, samples)
Z = np.random.normal(z_0 + z_1 * X + z_2 * Y, sigma_z)

data = pd.DataFrame([X,Y,Z], index = ['X','Y','Z'])
data.to_excel(os.path.join(r'C:\Users\Javier\Documents\MEGA\Universitattt\Master\Thesis\CDS_data\toy_data',\
				'toy_vstruc_normal.xlsx'))