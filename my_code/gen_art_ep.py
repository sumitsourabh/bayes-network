import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\epsilon_drawups'
    filename = r'artificial_eps.csv'

    length = 2001
    ep = pd.DataFrame(0.0, index = ['Russian.Fedn',2,3], columns = list(range(1,length)))

    delay = 3

    prob_1 = .06
    prob_given_1 = {2:.45, 3:.3}
    marg_prob = {2: .07, 3:.08}

    for c in ep.columns:
    	if np.random.rand() < prob_1:
    		ep.loc['Russian.Fedn',c] = 1.0

    		for i, p in prob_given_1.items():
    			if np.random.rand() < p:
    				# Apply delay
    				ep.loc[i,c + np.random.randint(0,min(4,length-c))] = 1.0
    				if ep.loc[i,c] != 1.0:
    					ep.loc[i,c] = .5

    	else:
    		for i, p in marg_prob.items():
    			if np.random.rand() < p:
    				ep.loc[i,c] = 1.0

    for c in ep.columns:
    	for i in range(1,min(4,length-c)):
    		if ep.loc['Russian.Fedn',c+i] == 1.0:
	    		if ep.loc['Russian.Fedn',c] == 0.0 and (ep.loc[2,c] == 1.0 or ep.loc[3,c] == 1.0):
	    			ep.loc['Russian.Fedn',c] = 0.5


    ep.to_csv(os.path.join(path,filename))