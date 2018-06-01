source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')

wind = 300
mov = 60

time_para = 20
delay = 1

percent = 0

# Firstly we get the data from the six periods

date1 = '2010-11-14' # to '2011-09-10'
date2 = '2011-09-10' # to '2012-07-06'
date3 = '2012-07-06' # to '2013-05-02'
date4 = '2013-05-02' # to '2014-02-26'
date5 = '2014-02-26' # to '2014-12-23'
date6 = '2014-10-24' # to '2015-08-14'

dates = list(date1,date2,date3,date4,date5,date6,'2015-08-14')

unpack[data_temp1, data_temp2] = read_dates(dates[[1]], dates[[2]], percent, wind, mov, time_para, delay)
unpack[data_temp3, data_temp4] = read_dates(dates[[3]], dates[[4]], percent, wind, mov, time_para, delay)
unpack[data_temp5, data_temp6] = read_dates(dates[[5]], dates[[6]], percent, wind, mov, time_para, delay)


Data = list(data_temp1,data_temp2,data_temp3,data_temp4,data_temp5,data_temp6)

path = 'C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Results/Probabilities'
folder = paste('time',as.character(time_para),'delay',as.character(delay),'per',as.character(percent),sep = '_')
dir.create(file.path(path, folder), showWarnings = FALSE)


#Define score, algorithm and number of bootstraps

used_score = 'bds'
RR = 1000
algorithm = 'hc'

timing = list()


for (i in 1:(length(dates)-1)){
  tm1 <- system.time(
    {
      data = read_one_date(dates[[i]], percent, wind, mov, time_para, delay)

      net = averaged.network(boot.strength(data = data, R = RR, algorithm = algorithm,
                                                       algorithm.args=list(score=used_score)))

      net = direct_net(net, used_score = used_score, Data = data)

      prob = matrix_prob(net, data)

      filename = paste(dates[i],dates[i+1],'prob','matrix.csv',sep = '_')

      write.csv(prob, file.path(path,folder,filename))
    })
  cmd = paste('net_', as.character(i),' = net', sep = '')
  eval(parse(text = cmd))
  t = as.list(tm1)$elapsed
  print(t)
  timing = append(timing,t)
}
timing = data.frame(timing)
colnames(timing) = 1:(length(dates)-1)
rownames(timing) = 'time'
write.csv(timing, file = file.path(path, folder, 'time_elapsed.csv'))
save(net_1,net_2,net_3,net_4,net_5,net_6,file = file.path(path, folder, 'networks.RData'))

