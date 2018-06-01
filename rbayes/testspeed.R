wind = 300
mov = 60

time_para = 15
delay = 1

percent = 5

date1 = "2012-07-06"
date2 = "2014-02-26"

unpack[Data15_1, Data15_2] = read_dates(date1, date2, percent, wind, mov, time_para, delay)

used_score = 'k2'
RR = 1000
algorithm = 'hc'

tm_hc <- system.time(
  {
    source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')
    pre_k2_hc_15_d1 = averaged.network(boot.strength(data = Data15_1, R = RR, algorithm = 'hc',
                                                        algorithm.args=list(score=used_score)))
    
  })

print('k2 hc')
print(as.list(tm_hc)$elapsed)


tabu_para = 50

tm_ta50 <- system.time(
  {
    source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')
    pre_k2_ta50_15_d1 = averaged.network(boot.strength(data = Data15_1, R = RR, algorithm = 'tabu',
                                         algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
    
  })

print('k2 tabu 50')
print(as.list(tm_ta50)$elapsed)

tabu_para = 100

tm_ta100 <- system.time(
  {
    source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')
    pre_k2_ta100_15_d1 = averaged.network(boot.strength(data = Data15_1, R = RR, algorithm = 'tabu',
                                          algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
    
  })

print('k2 tabu 100')
print(as.list(tm_ta100)$elapsed)


used_score = 'bds'

tm_hc_bds <- system.time(
  {
    source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')
    pre_bds_hc_15_d1 = averaged.network(boot.strength(data = Data15_1, R = RR, algorithm = 'hc',
                                                        algorithm.args=list(score=used_score)))
    
  })

print('bds hc')
print(as.list(tm_hc_bds)$elapsed)


tabu_para = 50

tm_ta50_bds <- system.time(
  {
    source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')
    pre_bds_ta50_15_d1 = averaged.network(boot.strength(data = Data15_1, R = RR, algorithm = 'tabu',
                                                       algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
    
  })

print('bds tabu 50')
print(as.list(tm_ta50_bds)$elapsed)

tabu_para = 100

tm_ta100_bds <- system.time(
  {
    source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')
    pre_bds_ta100_15_d1 = averaged.network(boot.strength(data = Data15_1, R = RR, algorithm = 'tabu',
                                                        algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
    
  })

print('bds tabu 100')
print(as.list(tm_ta100_bds)$elapsed)
