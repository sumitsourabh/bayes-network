import pandas as pd
import datetime as datetime
import numpy as np
import os

def select_counterparties(region,sector,rating,date_list):
	''' This function selects the counterparties according to the input arguments'''

	# DataFrame containing all the data for the desired counterparties
	counterparty_data = pd.DataFrame()
	#columns=['MTM_DATE','SHORTNAME','TICKER','SPREAD5Y','RECOVERY_RATE','SECTOR','REGION','COUNTRY','AV_RATING'])
	for date in date_list:
		print(date)
		filename = os.path.join(r'C:\Users\Javier\Documents\MEGA\Universitattt',
			'Master\Thesis\CDS_data\CDSdailydata', date+ '.csv')

		dfDay = pd.read_csv(filename, skiprows=0, usecols=['index','MTM_DATE','SHORTNAME',\
				'TICKER','TIER','CCY','DOCCLAUSE','SPREAD5Y','SECTOR','REGION','COUNTRY','AV_RATING','RECOVERY_RATE'])

		counterparties_df=dfDay[((dfDay['REGION']==region) & (dfDay['SECTOR']==sector) & (dfDay['AV_RATING']==rating))]
		counterparties_df=filter_data(counterparties_df)
		counterparties_df=counterparties_df[['MTM_DATE','SHORTNAME','TICKER','SPREAD5Y','RECOVERY_RATE','SECTOR','REGION','COUNTRY','AV_RATING']]
		
		#print(counterparties_df.shape[0])
		counterparty_data=counterparty_data.append(counterparties_df)

		# NOTE: This could be done at the end, outside the for loop
		spread=counterparty_data['SPREAD5Y']
		recovery=counterparty_data['RECOVERY_RATE']
		norm_spread=[s*0.6/(1-r) for (s,r) in zip (spread,recovery)]
		counterparty_data['SPREAD5Y']=spread
		counterparty_data['NORM_SPREAD']=norm_spread

	return(counterparty_data)

def select_counterparties_by_ctry(country):
	counterparty_data = pd.DataFrame()
	#columns=['MTM_DATE','SHORTNAME','TICKER','SPREAD5Y','RECOVERY_RATE','SECTOR','REGION','COUNTRY','AV_RATING'])
	for date in date_list:
		print(date)
		filename = os.path.join(r'C:\Users\Javier\Documents\MEGA\Universitattt',\
			'Master\Thesis\CDS_data\CDSdailydata', date+ '.csv')

		dfDay = pd.read_csv(filename, skiprows=0, usecols=['index','MTM_DATE','SHORTNAME',\
			'TICKER','TIER','CCY','DOCCLAUSE','SPREAD5Y','SECTOR','REGION','COUNTRY','AV_RATING','RECOVERY_RATE'])

		counterparties_df=dfDay[(dfDay['COUNTRY']==country)]
		counterparties_df=filter_data(counterparties_df)
		counterparties_df=counterparties_df[['MTM_DATE','SHORTNAME','TICKER','SPREAD5Y','RECOVERY_RATE','SECTOR','REGION','COUNTRY','AV_RATING']]

		#print(counterparties_df.shape[0])
		counterparty_data=counterparty_data.append(counterparties_df)

		# NOTE: This could be done at the end, outside the for loop
		spread=counterparty_data['SPREAD5Y']
		recovery=counterparty_data['RECOVERY_RATE']
		norm_spread=[s*0.6/(1-r) for (s,r) in zip (spread,recovery)]
		counterparty_data['SPREAD5Y']=spread
		counterparty_data['NORM_SPREAD']=norm_spread

	return(counterparty_data)

def select_counterparties_by_region_single(region,date_list_single):
	counterparty_data = pd.DataFrame()
	#columns=['MTM_DATE','SHORTNAME','TICKER','SPREAD5Y','RECOVERY_RATE','SECTOR','REGION','COUNTRY','AV_RATING'])

	for date in date_list_single:
		print(date)
		filename = os.path.join(r'C:\Users\Javier\Documents\MEGA\Universitattt',\
			'Master\Thesis\CDS_data\CDSdailydata', date+ '.csv')

		dfDay=pd.read_csv(filename, skiprows=0,usecols=['index','MTM_DATE','SHORTNAME',\
			'TICKER','TIER','CCY','DOCCLAUSE','SPREAD5Y','SECTOR','REGION','COUNTRY','AV_RATING','RECOVERY_RATE'])
		counterparties_df=dfDay[(dfDay['REGION']==region)]
		counterparties_df=filter_data(counterparties_df)
		counterparties_df=counterparties_df[['MTM_DATE','SHORTNAME','TICKER','SPREAD5Y','RECOVERY_RATE','SECTOR','REGION','COUNTRY','AV_RATING']]

		#print(counterparties_df.shape[0])
		counterparty_data=counterparty_data.append(counterparties_df)

		# NOTE: This could be done at the end, outside the for loop
		spread=counterparty_data['SPREAD5Y']
		recovery=counterparty_data['RECOVERY_RATE']
		norm_spread=[s*0.6/(1-r) for (s,r) in zip (spread,recovery)]
		counterparty_data['SPREAD5Y']=spread
		counterparty_data['NORM_SPREAD']=norm_spread

	return(counterparty_data)

def select_counterparties_byname(name,date_list):
	counterparty_data=pd.DataFrame()
	for date in date_list:
		print(date)
		filename = os.path.join(r'C:\Users\Javier\Documents\MEGA\Universitattt',\
			'Master\Thesis\CDS_data\CDSdailydata', date+ '.csv')

		dfDay=pd.read_csv(filename, skiprows=0, usecols=['index','MTM_DATE','SHORTNAME',\
			'TICKER','TIER','CCY','DOCCLAUSE','SPREAD5Y','SECTOR','REGION','COUNTRY','AV_RATING','RECOVERY_RATE'])

		counterparties_df=dfDay[(dfDay['SHORTNAME']==name)]#|(dfDay['COUNTRY']=='Russian Federation')]
		counterparties_df=filter_data(counterparties_df)			
		counterparties_df=counterparties_df[['MTM_DATE','SHORTNAME','TICKER','SPREAD5Y','RECOVERY_RATE','SECTOR','REGION','COUNTRY','AV_RATING']]
		#print(counterparties_df.shape[0])
		counterparty_data=counterparty_data.append(counterparties_df)

		# NOTE: This could be done at the end, outside the for loop
		spread=counterparty_data['SPREAD5Y']
		recovery=counterparty_data['RECOVERY_RATE']
		norm_spread=[s*0.6/(1-r) for (s,r) in zip (spread,recovery)]
		counterparty_data['SPREAD5Y']=spread
		counterparty_data['NORM_SPREAD']=norm_spread
	
	return(counterparty_data)

def merge_counterparty_data(counterparty_data,date_list):
	#
	df_merged = counterparty_data[(counterparty_data['MTM_DATE']==date_list[0])]
	df_merged.columns=[u'AV_RATING', u'COUNTRY', u'MTM_DATE', u'NORM_SPREAD ' + date_list[0],\
		u'RECOVERY_RATE ' + date_list[0], u'REGION', u'SECTOR', u'SHORTNAME', u'SPREAD5Y ' + date_list[0], u'TICKER']
	for date in date_list[1:]:
		print(date)
		#print(df_merged.shape)
		df = counterparty_data[(counterparty_data['MTM_DATE']==date)]
		df = df[['SHORTNAME','SPREAD5Y','RECOVERY_RATE','NORM_SPREAD']]
		df.columns = ['SHORTNAME', 'SPREAD5Y ' + date, 'RECOVERY_RATE' + date, 'NORM_SPREAD ' + date]
		df_merged = pd.merge(df, df_merged, how='inner',on=['SHORTNAME'])
	#filename=region+'_'+sector+'_'+rating+'_data.csv'
	#df_merged.to_csv(filename)
	return(df_merged)

def filter_data(dfTest):
	# Removes the None and NaN values from the data
	dfTest=dfTest[~(pd.isnull(dfTest['AV_RATING']))]
	dfTest=dfTest[~(pd.isnull(dfTest['SPREAD5Y']))]
	#Filtering extra names
	dfRus=dfTest[(dfTest['REGION']=='E.Eur')]
	dfNAmer=dfTest[(dfTest['REGION']=='N.Amer')]
	dfEurope=dfTest[(dfTest['REGION']=='Europe')]
	# Filters more things
	dfRus=dfRus[((dfRus['DOCCLAUSE']=='CR14')|(dfRus['DOCCLAUSE']=='CR')) & (dfRus['CCY'] == 'USD') & (dfRus['TIER']=='SNRFOR')]
	dfNAmer=dfNAmer[((dfNAmer['DOCCLAUSE']=='XR14')|(dfNAmer['DOCCLAUSE']=='XR')) & (dfNAmer['CCY'] == 'USD') & (dfNAmer['TIER']=='SNRFOR')]
	dfEurope=dfEurope[((dfEurope['DOCCLAUSE']=='CR14')|(dfEurope['DOCCLAUSE']=='CR')) & (dfEurope['CCY'] == 'EUR') & (dfEurope['TIER']=='SNRFOR') ]
	#dfEurope=dfEurope[((dfEurope['DOCCLAUSE']=='MM14')|(dfEurope['DOCCLAUSE']=='MM')) & (dfEurope['CCY'] == 'EUR') & (dfEurope['TIER']=='SNRFOR') ]
	frames = [dfRus, dfEurope, dfNAmer]
	dfTest = pd.concat(frames)
	#dfTest.reindex(np.random.permutation(dfTest.index))
	return(dfTest)

def create_numpy_array(entities_df, date_list):
	[row, col] = entities_df.shape
	entities_np = np.empty([row , (col/3)-1], dtype=object)
	col_list = ['NORM_SPREAD ' + date for date in date_list]
	entities_list = list(entities_df['SHORTNAME'])
	entities_np[:, 0] = entities_list

	for i in range(row):
		entities_np[i, 1:] = 10000 * np.ravel(entities_df[(entities_df['SHORTNAME'] ==
														entities_list[i])][col_list])
	print(entities_np )       
	# np.save('/Users/Sumit/Dropbox/Russianess/data/entities_data_germany', entities_np)