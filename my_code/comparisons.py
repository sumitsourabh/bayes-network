import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import utilities as ut
from scipy.stats import normaltest
import datetime as dt
import seaborn as sns
import networkx as nx

def comp_perc_cap(per1, per2, delay):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Capital_periods'
    folder1 = r'results_delay_'+str(delay)+'_per_'+str(per1)
    folder2 = r'results_delay_'+str(delay)+'_per_'+str(per2)

    no_con_1 = pd.read_csv(os.path.join(path,folder1,\
        'No_contagion_'+str(delay)+'_'+str(per1)+'.csv'),index_col = 0)
    con_1 = pd.read_csv(os.path.join(path,folder1,\
        'Contagion_'+str(delay)+'_'+str(per1)+'.csv'),index_col = 0)
    no_con_2 = pd.read_csv(os.path.join(path,folder2,\
        'No_contagion_'+str(delay)+'_'+str(per2)+'.csv'),index_col = 0)
    con_2 = pd.read_csv(os.path.join(path,folder2,\
        'Contagion_'+str(delay)+'_'+str(per2)+'.csv'),index_col = 0)

    comp_cont = (con_2-con_1)/con_1*100

    to_perc = lambda x: "{0:.2f}".format(x)

    for c in comp_cont.columns:
        for i in comp_cont.index:
            comp_cont.loc[i,c] = to_perc(comp_cont.loc[i,c])

    folder_save = r'comparisons'

    if not os.path.isdir(os.path.join(path,folder_save)):
        os.makedirs(os.path.join(path,folder_save))

    comp_cont.to_csv(os.path.join(path,folder_save,\
            'contagion_'+str(per1)+'_to_'+str(per2)+'.csv'))

def comp_prob_def(per1, per2, delay, counterparty):
    path = r'C:\Users\Javier\Documents\MEGA\Thesis\Results\Probabilities'
    files_1 = sorted([f for f in os.listdir(path) if \
            f.endswith('_delay_'+str(delay)+'_per_'+str(per1))])
    files_2 = sorted([f for f in os.listdir(path) if \
            f.endswith('_delay_'+str(delay)+'_per_'+str(per2))])

    list_df = []

    if len(files_1)==len(files_2):
        prob = sorted([l for l in os.listdir(files_1[0]) if l.endswith('prob_matrix.csv')])
        for i in range(len(files_1)):
            for f in prob:
                df1 = pd.read_csv(os.path.join(path,files_1[i],f))
                RUS1 = df1.transpose()['RUSSIA']
                df2 = pd.read_csv(os.path.join(path,files_2[i],f))
                RUS2 = df2.transpose()['RUSSIA']
                list_df.append((df2-df1)/df1)

    return()

    else:
        print('Error: not same number of files for the percentages in \n' + path)

def comp_prob():
    path_countryrank = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\Ioannis'
    port_a = pd.read_csv(os.path.join(path_countryrank), 'PortfolioA.csv')
    dic = {'Government of the Russian Federation':'RUSSIA','Oil Transporting Jt Stk Co Transneft':'AKT',
        'Vnesheconombank':'BKECON','Bank of Moscow OJSC':'BOM','City Moscow':'CITMOS','GAZPROM PJSC':'GAZPRU',
        'JSC Gazprom Neft':'GAZPRU.Gneft','LUKOIL PJSC':'LUKOIL','Mobile Telesystems':'MBT',
        'MDM Bank JSC':'MDMOJC','ALROSA Company Ltd':'ALROSA','Rosneftegaz OJSC':'ROSNEF',
        'Jt Stk Co Russian Standard Bk':'RSBZAO','Joint-stock company Russian Agricultural Bank':'RUSAGB',
        'JSC Russian Railways':'RUSRAI','Sberbank of Russia':'SBERBANK','VimpelCom Ltd.':'VIP','VTB Bank (public joint-stock company)':'VTB'}

    for i in port_a.index:
        port_a.loc[i,'Ticker'] = dic[i]

    path = r'C:\Users\Javier\Documents\MEGA\Thesis\CDS_data\factor_model\compare_period'
    for folder in os.listdir(path):
        aux = port_a[['Ticker','gamma']]
        for file in [f for f in os.listdir(os.path.join(path,folder)) if f.startswith('portfolio')]:
            df = pd.read_csv(os.path.join(path,folder,file),index_col = 0)
            print(file)
            num = [s for s in file.replace('.csv','').split('_') if s.isdigit()]
            df.rename(columns = {'gamma':'gamma '+str(num[1])+' days'}, inplace = True)
            aux = pd.concat([aux, df['gamma '+str(num[1])+' days']], axis = 1)
        aux.sort_values(by = 'gamma', ascending = False, inplace = True)
        aux.loc['Mean','Ticker'] = '-'
        aux.iloc[-1,1:] = np.mean(aux)
        aux.to_csv(os.path.join(path,folder, 'comparisons_'+str(num[0]+'.csv')))
