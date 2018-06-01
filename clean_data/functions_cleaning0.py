import numpy as np 
import pandas as pd 
import os
import requests

def clean_one_day(df):
	''' This function receives the dataframe of the data of CDS from one day
		and returns the clean data'''
	df.drop(['CONTRIBUTOR', 'SPREAD6M', 'SPREAD1Y', 'SPREAD2Y','DATA_RATING',
	        'SPREAD3Y', 'SPREAD4Y','SPREAD7Y', 'SPREAD10Y',
	        'SPREAD15Y', 'SPREAD20Y', 'SPREAD30Y'],axis=1, inplace = True)
	# df.index = range(len(df.index))
	Ticker_wanted = ['AKT','BKECON','BOM','CITMOS','GAZPRU','GAZPRU-Gneft','LUKOIL',
	                 'MBT','MDMOJC','OPENJOCH','ROSNEF','RSBZAO','RUSAGB','RUSRAI',
	                 'RUSSIA','SBERBANK','VIP','VTB','EVRGSA','INTNED']
	Euro_zone = ['Austria','Belgium','Cyprus','Estonia','Finland','France','Germany',
	'Greece','Ireland','Italy','Luxembourg','Malta','Netherlands','Portugal',
	'Slovakia','Slovenia','Spain']

	df = df[df.TICKER.isin(Ticker_wanted)]
	df = df[df.CCY.isin(['EUR','USD'])]
	df = df[df.TIER == 'SNRFOR']
	df2 = pd.DataFrame(data = [], columns = df.columns)

	score_Europe = {'MM14':1, 'MM':2, 'MR14':3, 'MR':4, 'CR14':5, 'CR':6}                

	for i in df.index:
		# If we already read a quote from the counterparty
	    if df.TICKER[i] in df2.TICKER.unique():
	        j = df2[df2.TICKER == df.TICKER[i]].index[0]
	        if df.SECTOR[i] == 'Government':
	            if df.COUNTRY[i] in Euro_zone and df.CCY[i] == 'EUR':
	                if df.DOCCLAUSE[i] == 'CR14' and df2.DOCCLAUSE[j] == 'CR':
	                    df2.loc[j,:] = df.loc[i,:]
	            if df.COUNTRY[i] not in Euro_zone and df.COUNTRY[i] != 'Australia' and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'CR14' and df2.DOCCLAUSE[j] == 'CR':
	                    df2.loc[i,:] = df.loc[i,:]
	            if df.COUNTRY[i] == 'Australia' and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'MR14'  and df2.DOCCLAUSE[j] == 'MR':
	                    df2.loc[i,:] = df.loc[i,:]
	        else:
	            if df.REGION[i] in ['Europe', 'E.Eur'] and df.CCY[i] == 'EUR':
	                if score_Europe[df.DOCCLAUSE[i]] < score_Europe[df2.DOCCLAUSE[j]]:
	                    df2.loc[j,:] = df.loc[i,:]
	            elif 'Amer' in df.REGION[i] and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'XR14'  and df2.DOCCLAUSE[j] == 'XR':
	                    df2.loc[j,:] = df.loc[i,:]
	            elif df.REGION[i] == 'Oceania' and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'MR14'  and df2.DOCCLAUSE[j] == 'MR':
	                    df2.loc[j,:] = df.loc[i,:]
	            elif df.REGION[i] not in ['Europe', 'E.Eur', 'Oceania'] and 'Amer' not in df.REGION[i] and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'CR14'   and df2.DOCCLAUSE[j] == 'CR':
	                    df2.loc[j,:] = df.loc[i,:]
	                    
	    # In case it is new.
	    else:
	        if df.SECTOR[i] == 'Government':
	            if df.COUNTRY[i] in Euro_zone and df.CCY[i] == 'EUR':
	                if df.DOCCLAUSE[i] == 'CR14':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'CR':
	                    df2.loc[i,:] = df.loc[i,:]
	            if df.COUNTRY[i] not in Euro_zone and df.COUNTRY[i] != 'Australia' and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'CR14':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'CR':
	                    df2.loc[i,:] = df.loc[i,:]
	            if df.COUNTRY[i] == 'Australia' and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'MR14':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'MR':
	                    df2.loc[i,:] = df.loc[i,:]
	        else:
	            if df.REGION[i] in ['Europe', 'E.Eur'] and df.CCY[i] == 'EUR':
	                if df.DOCCLAUSE[i] == 'MM14':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'MM':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'MR14':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'MR':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'CR14':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'CR':
	                    df2.loc[i,:] = df.loc[i,:]
	            elif 'Amer' in df.REGION[i] and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'XR14':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'XR':
	                    df2.loc[i,:] = df.loc[i,:]
	            elif df.REGION[i] == 'Oceania' and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'MR14':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'MR':
	                    df2.loc[i,:] = df.loc[i,:]
	            elif df.REGION[i] not in ['Europe', 'E.Eur', 'Oceania'] and 'Amer' not in df.REGION[i] and df.CCY[i] == 'USD':
	                if df.DOCCLAUSE[i] == 'CR14':
	                    df2.loc[i,:] = df.loc[i,:]
	                elif df.DOCCLAUSE[i] == 'CR':
	                    df2.loc[i,:] = df.loc[i,:]

	df2.loc[:,'NORMSPREAD5Y'] = df2.SPREAD5Y * 0.6/(1-df2.RECOVERY_RATE)

	return df2 

def add_to_history(hist, df):
    dat = df.loc[df.index[0],'MTM_DATE']
    for i in df.index:
        hist.loc[df.TICKER[i], dat] = df.NORMSPREAD5Y[i]

    hist.fillna(0.0)

    return hist