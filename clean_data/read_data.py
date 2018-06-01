import requests
import os
import datetime as dt

url = r'http://www.gmdb.intranet/data/markit/same_day_cds_london/cds/generic/same_day_cds_london_cds_generic_'

day = dt.datetime(2009,6,16).date()
d = dt.timedelta(days = 1)
num = '5'

today = dt.datetime.today().date()

change_number = dt.datetime(2010,9,14).date()
change_format = dt.datetime(2014,1,14).date()

path = r'\\ad.ing.net\fm\P\UD\313201\VN77VI\Home\My Documents\Data'

os.path.join(path,'cds_' + str(day).replace('-','') + form)

form = r'.csv.gz'

while day <= change_number:
	url_file = url + num + '_' + str(day).replace('-','') + form	
	## Reading part 
	r = requests.get(url_file, allow_redirects=True)
	open(os.path.join(path,'cds_' + str(day).replace('-','') + form), 'wb').write(r.content)
	day += d

num = '6'

while day <= change_format:
	url_file = url + num + '_' + str(day).replace('-','') + form 
	## Reading part
	r = requests.get(url_file, allow_redirects=True)
	open(os.path.join(path,'cds_' + str(day).replace('-','') + form), 'wb').write(r.content)	
	day += d

form = '.csv'

while day <= today:
	url_file = url + num + '_' + str(day).replace('-','') + form 	
	## Reading part
	r = requests.get(url_file, allow_redirects=True)
	open(os.path.join(path,'cds_' + str(day).replace('-','') + form), 'wb').write(r.content)
	day += d


