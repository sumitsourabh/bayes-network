import pandas as pd
import os
import utilities as ut

directory = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Probabilities'
score = 'bds'
filename = 'adjmat_' + score + '_hc_200_100.csv'
df = pd.read_csv(os.path.join(directory, filename), index_col = 0)

df = df + df.transpose()

ut.matrix_as_table_rot_names(df)