import numpy as np 
import pandas as pd 
import os
import time
import datetime as dt

def clean_one_day(df):

    start = time.time()

    ''' This function receives the dataframe of the data of CDS from one day
        and returns the clean data'''
    # df.index = range(len(df.index))
    # Ticker_wanted = ['AKT','BKECON','BOM','CITMOS','GAZPRU','GAZPRU-Gneft','LUKOIL',
    #                  'MBT','MDMOJC','OPENJOCH','ROSNEF','RSBZAO','RUSAGB','RUSRAI',
    #                  'RUSSIA','SBERBANK','VIP','VTB','EVRGSA','INTNED']

    # df = df[df.Ticker.isin(Ticker_wanted)]

    Euro_zone = ['Austria','Belgium','Cyprus','Estonia','Finland','France','Germany',
    'Greece','Ireland','Italy','Luxembourg','Malta','Netherlands','Portugal',
    'Slovakia','Slovenia','Spain']

    df = df[df.Ccy.isin(['EUR','USD'])]
    df = df[df.Tier == 'SNRFOR']
    df2 = pd.DataFrame(data = [], columns = df.columns)

    score_Europe = {'MM14':1, 'MM':2, 'MR14':3, 'MR':4, 'CR14':5, 'CR':6}

    ## Let us separate in different regions:
    df_Government = df[df.Sector == 'Government']
    df_NoGov = df[df.Sector != 'Government']

    df_Euro = df_NoGov[df_NoGov.Region.isin(['Europe','E.Eur'])]
    df_Othe = df_NoGov.drop(df_Euro.index, axis = 0)
    df_Euro = df_Euro[df_Euro.Ccy == 'EUR']

    df_Amer = df_NoGov[df_NoGov.Region.isin(['N.Amer','Lat.Amer'])]
    df_Othe = df_Othe.drop(df_Amer.index, axis = 0)
    df_Amer = df_Amer[df_Amer.Ccy == 'USD']

    df_Aust = df_NoGov[df_NoGov.Region == 'Oceania']
    df_Othe = df_Othe.drop(df_Aust.index, axis = 0)
    df_Aust = df_Aust[df_Aust.Ccy == 'USD']
    df_Othe = df_Othe[df_Othe.Ccy == 'USD']


    # Governments

    df_Gov_Eur = df_Government[df_Government.Country.isin(Euro_zone)]
    df_Gov_Rest = df_Government.drop(df_Gov_Eur.index, axis = 0)
    df_Gov_Rest = df_Gov_Rest[df_Gov_Rest.Ccy == 'USD']
    df_Gov_Eur = df_Gov_Eur[df_Gov_Eur.Ccy == 'EUR']


    # Euro Zone Government
    for u in df_Gov_Eur.Ticker.unique():
        df_aux = df_Gov_Eur[df_Gov_Eur.Ticker == u]

        if not df_aux[df_aux.DocClause == 'CR14'].empty:
            df_Gov_Eur = df_Gov_Eur.drop(df_aux[df_aux.DocClause != 'CR14'].index, axis = 0)
        else:
            df_Gov_Eur = df_Gov_Eur.drop(df_aux[df_aux.DocClause != 'CR'].index, axis = 0)


    # Australian Government
    df_Gov_Aus = df_Gov_Rest[df_Gov_Rest.Country == 'Australia']
    for u in df_Gov_Aus.Ticker.unique():
        df_aux = df_Gov_Aus[df_Gov_Aus.Ticker == u]

        if not df_aux[df_aux.DocClause == 'MR14'].empty:
            df_Gov_Aus = df_Gov_Aus.drop(df_aux[df_aux.DocClause != 'MR14'].index, axis = 0)
        else:
            df_Gov_Aus = df_Gov_Aus.drop(df_aux[df_aux.DocClause != 'MR'].index, axis = 0)


    # Rest of the Governments
    df_Gov_Rest = df_Gov_Rest[df_Gov_Rest.Country != 'Australia']
    for u in df_Gov_Rest.Ticker.unique():
        df_aux = df_Gov_Rest[df_Gov_Rest.Ticker == u]

        if not df_aux[df_aux.DocClause == 'CR14'].empty:
            df_Gov_Rest = df_Gov_Rest.drop(df_aux[df_aux.DocClause != 'CR14'].index, axis = 0)
        else:
            df_Gov_Rest = df_Gov_Rest.drop(df_aux[df_aux.DocClause != 'CR'].index, axis = 0)

    # No Governments

    # Euro Zone
    for u in df_Euro.Ticker.unique():
        df_aux = df_Euro[df_Euro.Ticker == u]

        if not df_aux[df_aux.DocClause == 'MM14'].empty:
            df_Euro = df_Euro.drop(df_aux[df_aux.DocClause != 'MM14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MM'].empty:
            df_Euro = df_Euro.drop(df_aux[df_aux.DocClause != 'MM'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MR14'].empty:
            df_Euro = df_Euro.drop(df_aux[df_aux.DocClause != 'MR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MR'].empty:
            df_Euro = df_Euro.drop(df_aux[df_aux.DocClause != 'MR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'CR14'].empty:
            df_Euro = df_Euro.drop(df_aux[df_aux.DocClause != 'CR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'CR'].empty:
            df_Euro = df_Euro.drop(df_aux[df_aux.DocClause != 'CR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'XR14'].empty:
            df_Euro = df_Euro.drop(df_aux[df_aux.DocClause != 'XR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'XR'].empty:
            df_Euro = df_Euro.drop(df_aux[df_aux.DocClause != 'XR'].index, axis = 0)


    # America
    for u in df_Amer.Ticker.unique():
        df_aux = df_Amer[df_Amer.Ticker == u]

        if not df_aux[df_aux.DocClause == 'XR14'].empty:
            df_Amer = df_Amer.drop(df_aux[df_aux.DocClause != 'XR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'XR'].empty:
            df_Amer = df_Amer.drop(df_aux[df_aux.DocClause != 'XR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'CR14'].empty:
            df_Amer = df_Amer.drop(df_aux[df_aux.DocClause != 'CR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'CR'].empty:
            df_Amer = df_Amer.drop(df_aux[df_aux.DocClause != 'CR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MR14'].empty:
            df_Amer = df_Amer.drop(df_aux[df_aux.DocClause != 'MR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MR'].empty:
            df_Amer = df_Amer.drop(df_aux[df_aux.DocClause != 'MR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MM14'].empty:
            df_Amer = df_Amer.drop(df_aux[df_aux.DocClause != 'MM14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MM'].empty:
            df_Amer = df_Amer.drop(df_aux[df_aux.DocClause != 'MM'].index, axis = 0)


    # Oceania
    for u in df_Aust.Ticker.unique():
        df_aux = df_Aust[df_Aust.Ticker == u]

        if not df_aux[df_aux.DocClause == 'MR14'].empty:
            df_Aust = df_Aust.drop(df_aux[df_aux.DocClause != 'MR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MR'].empty:
            df_Aust = df_Aust.drop(df_aux[df_aux.DocClause != 'MR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'CR14'].empty:
            df_Aust = df_Aust.drop(df_aux[df_aux.DocClause != 'CR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'CR'].empty:
            df_Aust = df_Aust.drop(df_aux[df_aux.DocClause != 'CR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'XR14'].empty:
            df_Aust = df_Aust.drop(df_aux[df_aux.DocClause != 'XR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'XR'].empty:
            df_Aust = df_Aust.drop(df_aux[df_aux.DocClause != 'XR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MM14'].empty:
            df_Aust = df_Aust.drop(df_aux[df_aux.DocClause != 'MM14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MM'].empty:
            df_Aust = df_Aust.drop(df_aux[df_aux.DocClause != 'MM'].index, axis = 0)

    # Rest (Including Asia because it is the same procedure)
    for u in df_Othe.Ticker.unique():
        df_aux = df_Othe[df_Othe.Ticker == u]

        if not df_aux[df_aux.DocClause == 'CR14'].empty:
            df_Othe = df_Othe.drop(df_aux[df_aux.DocClause != 'CR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'CR'].empty:
            df_Othe = df_Othe.drop(df_aux[df_aux.DocClause != 'CR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'XR14'].empty:
            df_Othe = df_Othe.drop(df_aux[df_aux.DocClause != 'XR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'XR'].empty:
            df_Othe = df_Othe.drop(df_aux[df_aux.DocClause != 'XR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MR14'].empty:
            df_Othe = df_Othe.drop(df_aux[df_aux.DocClause != 'MR14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MR'].empty:
            df_Othe = df_Othe.drop(df_aux[df_aux.DocClause != 'MR'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MM14'].empty:
            df_Othe = df_Othe.drop(df_aux[df_aux.DocClause != 'MM14'].index, axis = 0)
        elif not df_aux[df_aux.DocClause == 'MM'].empty:
            df_Othe = df_Othe.drop(df_aux[df_aux.DocClause != 'MM'].index, axis = 0)

    end = time.time()

    print(end-start)

    return pd.concat([df_Gov_Eur, df_Gov_Aus, df_Gov_Rest, df_Euro, df_Amer, df_Aust, df_Othe], axis = 0)


def add_to_history(hist, df):
    dat = df.loc[df.index[0],'MTM_DATE']
    for i in df.index:
        hist.loc[df.Ticker[i], dat] = df.NORMSPREAD5Y[i]

    hist.fillna(0.0)

    return hist


def write_data(df, name):
	path = r'\\ad.ing.net\fm\P\UD\313201\VN77VI\Home\My Documents\Data_processed'
	df.to_csv(os.path.join(path,name))

def date_from_file(file):
	date_int = ''.join([s for s in file if s.isdigit()])
	year = int(date_int[:4])
	month = int(date_int[4:6])
	day = int(date_int[6:])
	return dt.datetime(year,month, day).date()

def process_data():
	path_data = r'\\ad.ing.net\fm\P\UD\313201\VN77VI\Home\My Documents\Data'
	path_processed = r'\\ad.ing.net\fm\P\UD\313201\VN77VI\Home\My Documents\Data_processed'
	files_processed = os.listdir(path_processed)
	date_of_change = dt.datetime(2010,9,13).date()
	files_to_proc = [file for file in os.listdir(path_data) if date_from_file(file) > date_of_change and file not in files_processed]
	for filename in files_to_proc:
		df = pd.read_csv(os.path.join(path,filename), header = 1)
		df2 = clean_one_day(df)
		write_data(df2,filename)

if __name__ == '__main__':
	process_data()


	# OBSOLETE CODE
	#
    # for i in df.index:
    #     # If we already read a quote from the counterparty
    #     if df.Ticker[i] in df2.Ticker.unique():
    #         j = df2[df2.Ticker == df.Ticker[i]].index[0]
    #         if df.Sector[i] == 'Government':
    #             if df.Country[i] in Euro_zone and df.Ccy[i] == 'EUR':
    #                 if df.DocClause[i] == 'CR14' and df2.DocClause[j] == 'CR':
    #                     df2.loc[j,:] = df.loc[i,:]
    #             if df.Country[i] not in Euro_zone and df.Country[i] != 'Australia' and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'CR14' and df2.DocClause[j] == 'CR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #             if df.Country[i] == 'Australia' and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'MR14'  and df2.DocClause[j] == 'MR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #         else:
    #             if df.Country[i] in Euro_zone and df.Ccy[i] == 'EUR':
    #                 if df.DocClause[i] == 'MM14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'MM':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'MR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'MR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'CR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'CR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #             elif 'Amer' in df.Region[i] and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'XR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'XR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #             elif df.Region[i] == 'Oceania' and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'MR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'MR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #             elif df.Region[i] not in ['Europe', 'E.Eur', 'Oceania'] and 'Amer' not in df.Region[i] and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'CR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'CR':
    #                     df2.loc[i,:] = df.loc[i,:]
                        
    #     # In case it is new.
    #     else:
    #         if df.Sector[i] == 'Government':
    #             if df.Country[i] in Euro_zone and df.Ccy[i] == 'EUR':
    #                 if df.DocClause[i] == 'CR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'CR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #             if df.Country[i] not in Euro_zone and df.Country[i] != 'Australia' and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'CR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'CR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #             if df.Country[i] == 'Australia' and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'MR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'MR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #         else:
    #             if df.Country[i] in Euro_zone and df.Ccy[i] == 'EUR':
    #                 if df.DocClause[i] == 'MM14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'MM':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'MR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'MR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'CR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'CR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #             elif 'Amer' in df.Region[i] and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'XR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'XR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #             elif df.Region[i] == 'Oceania' and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'MR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'MR':
    #                     df2.loc[i,:] = df.loc[i,:]
    #             elif df.Region[i] not in ['Europe', 'E.Eur', 'Oceania'] and 'Amer' not in df.Region[i] and df.Ccy[i] == 'USD':
    #                 if df.DocClause[i] == 'CR14':
    #                     df2.loc[i,:] = df.loc[i,:]
    #                 elif df.DocClause[i] == 'CR':
    #                     df2.loc[i,:] = df.loc[i,:]

    # df2.Spread5y = df.Spread5y.apply(lambda x : float(str(x).replace(r'%','')))
    # df2.Recovery = df.Recovery.apply(lambda x : float(str(x).replace(r'%','')))

    # df2.loc[:,'Normspread5y'] = df2.Spread5y * 0.6/(1-df2.Recovery)

    # df2.drop(['Timezone','RedCode','Tier','DocClause','DataRating'],axis=1, inplace = True)

    # return df2 
