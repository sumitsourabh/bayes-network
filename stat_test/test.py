import pandas as pd
import numpy as np
import sklearn.metrics as skm
import scipy.stats as scs


def categories(series):
    return range(int(series.min()), int(series.max()) + 1)


def chi_square_of_df_cols(df, col1, col2):
    df_col1, df_col2 = df[col1], df[col2]

    result = [[sum((df_col1 == cat1) & (df_col2 == cat2))
               for cat2 in categories(df_col2)]
              for cat1 in categories(df_col1)]

    return scs.chi2_contingency(result)


test_df = pd.read_excel('test_data.xlsx')
print(chi_square_of_df_cols(test_df, 'var1', 'var2'))

df2 = pd.DataFrame(data=np.random.rand(1000,2)>0.5, columns = ['var1', 'var2'])

print(chi_square_of_df_cols(df2, 'var1', 'var2'))

print(skm.mutual_info_score(test_df.loc[1:,'var1'],test_df.loc[1:,'var2']))

print(skm.mutual_info_score(df2.loc[1:,'var1'],df2.loc[1:,'var2']))
