import pandas as pd
import os
import utilities as ut

directory = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Probabilities'
score = 'bds'
method = 'mle'
filename = method + '_' + score + '_hc_200_100.csv'
df = pd.read_csv(os.path.join(directory, filename), index_col = 0)

ut.prob_as_table(df, False)