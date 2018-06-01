
source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')

wind = 300
mov = 60

time_para = 10
delay = 1

percent = 5

date1 = "2012-07-06"
date2 = "2014-02-26"

unpack[Data10_1_d1, Data10_2_d1] = read_dates(date1, date2, percent, wind, mov, time_para, delay)

used_score = 'k2'
RR = 1000
algorithm = 'hc'

time_matrix = matrix(0,4,3)
rownames(time_matrix) = list('k2_pre', 'bds_pre', 'k2_cri', 'bds_cri')
colnames(time_matrix) = list('hc', 'tabu_50', 'tabu_100')

period = 'pre'
algorithm = 'hc'

# HC WITH K2 PRE
tm_k2_hc_pre <- system.time(
  {
    pre_k2_hc_10_d1 = averaged.network(boot.strength(data = Data10_1_d1, R = RR, algorithm = algorithm,
                                                     algorithm.args=list(score=used_score)))
  })

pre_k2_hc_10_d1 = direct_net(pre_k2_hc_10_d1, used_score, Data10_1_d1)

bayes_pre_k2_hc_10_d1 = compute_prob_v2(pre_k2_hc_10_d1, 'RUSSIA', Data10_1_d1)

print('k2 hc pre')
print(as.list(tm_k2_hc_pre)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_k2_hc_pre)$elapsed

# TABU 50 WITH K2 PRE

algorithm = 'tabu'
tabu_para = 50

tm_k2_ta50_pre <- system.time(
  {
    pre_k2_ta50_10_d1 = averaged.network(boot.strength(data = Data10_1_d1, R = RR, algorithm = algorithm,
                                                       algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
  })
pre_k2_ta50_10_d1 = direct_net(pre_k2_ta50_10_d1, used_score, Data10_1_d1)

bayes_pre_k2_ta50_10_d1 = compute_prob_v2(pre_k2_ta50_10_d1, 'RUSSIA', Data10_1_d1)


print('k2 tabu 50 pre')
print(as.list(tm_k2_ta50_pre)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_k2_ta50_pre)$elapsed


# TABU 100 WITH K2 PRE

tabu_para = 100

tm_k2_ta100_pre <- system.time(
  {
    pre_k2_ta100_10_d1 = averaged.network(boot.strength(data = Data10_1_d1, R = RR, algorithm = algorithm,
                                                        algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
    
  })
pre_k2_ta100_10_d1 = direct_net(pre_k2_ta100_10_d1, used_score, Data10_1_d1)

bayes_pre_k2_ta100_10_d1 = compute_prob_v2(pre_k2_ta100_10_d1, 'RUSSIA', Data10_1_d1)

print('k2 tabu 100 pre')
print(as.list(tm_k2_ta100_pre)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_k2_ta100_pre)$elapsed

period = 'cri'
algorithm = 'hc'

# HC WITH K2 CRI
tm_k2_hc_cri <- system.time(
  {
    cri_k2_hc_10_d1 = averaged.network(boot.strength(data = Data10_2_d1, R = RR, algorithm = algorithm,
                                                     algorithm.args=list(score=used_score)))
    
  })
print('aaaa')
cri_k2_hc_10_d1 = direct_net(cri_k2_hc_10_d1, used_score, Data10_2_d1)

bayes_cri_k2_hc_10_d1 = compute_prob_v2(cri_k2_hc_10_d1, 'RUSSIA', Data10_2_d1)

print('k2 hc cri')
print(as.list(tm_k2_hc_cri)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_k2_hc_cri)$elapsed

# TABU 50 WITH K2 cri

algorithm = 'tabu'
tabu_para = 50

tm_k2_ta50_cri <- system.time(
  {
    cri_k2_ta50_10_d1 = averaged.network(boot.strength(data = Data10_2_d1, R = RR, algorithm = algorithm,
                                                       algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
    
  })
cri_k2_ta50_10_d1 = direct_net(cri_k2_ta50_10_d1, used_score, Data10_2_d1)

bayes_cri_k2_ta50_10_d1 = compute_prob_v2(cri_k2_ta50_10_d1, 'RUSSIA', Data10_2_d1)

print('k2 tabu 50 cri')
print(as.list(tm_k2_ta50_cri)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_k2_ta50_cri)$elapsed


# TABU 100 WITH K2 cri

tabu_para = 100

tm_k2_ta100_cri <- system.time(
  {
    cri_k2_ta100_10_d1 = averaged.network(boot.strength(data = Data10_2_d1, R = RR, algorithm = algorithm,
                                                        algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
    
  })

cri_k2_ta100_10_d1 = direct_net(cri_k2_ta100_10_d1, used_score, Data10_2_d1)

bayes_cri_k2_ta100_10_d1 = compute_prob_v2(cri_k2_ta100_10_d1, 'RUSSIA', Data10_2_d1)

print('k2 tabu 100 cri')
print(as.list(tm_k2_ta100_cri)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_k2_ta100_cri)$elapsed


# CHANGE OF SCORE
# HC WITH BDS PRE

used_score = 'bds'
period = 'pre'
algorithm = 'hc'

# HC WITH BDS PRE
tm_bds_hc_pre <- system.time(
  {
    pre_bds_hc_10_d1 = averaged.network(boot.strength(data = Data10_1_d1, R = RR, algorithm = algorithm,
                                                      algorithm.args=list(score=used_score)))
  })

pre_bds_hc_10_d1 = direct_net(pre_bds_hc_10_d1, used_score, Data10_1_d1)

bayes_pre_bds_hc_10_d1 = compute_prob_v2(pre_bds_hc_10_d1, 'RUSSIA', Data10_1_d1)

print('bds hc pre')
print(as.list(tm_bds_hc_pre)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_bds_hc_pre)$elapsed

# TABU 50 WITH bds PRE

algorithm = 'tabu'
tabu_para = 50

tm_bds_ta50_pre <- system.time(
  {
    pre_bds_ta50_10_d1 = averaged.network(boot.strength(data = Data10_1_d1, R = RR, algorithm = algorithm,
                                                        algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
  })

pre_bds_ta50_10_d1 = direct_net(pre_bds_ta50_10_d1, used_score, Data10_1_d1)

bayes_pre_bds_ta50_10_d1 = compute_prob_v2(pre_bds_ta50_10_d1, 'RUSSIA', Data10_1_d1)

print('bds tabu 50 pre')
print(as.list(tm_bds_ta50_pre)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_bds_ta50_pre)$elapsed


# TABU 100 WITH bds PRE

tabu_para = 100

tm_bds_ta100_pre <- system.time(
  {
    pre_bds_ta100_10_d1 = averaged.network(boot.strength(data = Data10_1_d1, R = RR, algorithm = algorithm,
                                                         algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
  })

pre_bds_ta100_10_d1 = direct_net(pre_bds_ta100_10_d1, used_score, Data10_1_d1)

bayes_pre_bds_ta100_10_d1 = compute_prob_v2(pre_bds_ta100_10_d1, 'RUSSIA', Data10_1_d1)

print('bds tabu 100 pre')
print(as.list(tm_bds_ta100_pre)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_bds_ta100_pre)$elapsed

period = 'cri'

algorithm = 'hc'

# HC WITH bds CRI
tm_bds_hc_cri <- system.time(
  {
    cri_bds_hc_10_d1 = averaged.network(boot.strength(data = Data10_2_d1, R = RR, algorithm = algorithm,
                                                      algorithm.args=list(score=used_score)))
  })

cri_bds_hc_10_d1 = direct_net(cri_bds_hc_10_d1, used_score, Data10_2_d1)

bayes_cri_bds_hc_10_d1 = compute_prob_v2(cri_bds_hc_10_d1, 'RUSSIA', Data10_2_d1)

print('bds hc cri')
print(as.list(tm_bds_hc_cri)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_bds_hc_cri)$elapsed

# TABU 50 WITH bds cri


algorithm = 'tabu'
tabu_para = 50

tm_bds_ta50_cri <- system.time(
  {
    cri_bds_ta50_10_d1 = averaged.network(boot.strength(data = Data10_2_d1, R = RR, algorithm = algorithm,
                                                        algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
  })

cri_bds_ta50_10_d1 = direct_net(cri_bds_ta50_10_d1, used_score, Data10_2_d1)

bayes_cri_bds_ta50_10_d1 = compute_prob_v2(cri_bds_ta50_10_d1, 'RUSSIA', Data10_2_d1)

print('bds tabu 50 cri')
print(as.list(tm_bds_ta50_cri)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_bds_ta50_cri)$elapsed


# TABU 100 WITH bds cri

tabu_para = 100

tm_bds_ta100_cri <- system.time(
  {
    cri_bds_ta100_10_d1 = averaged.network(boot.strength(data = Data10_2_d1, R = RR, algorithm = algorithm,
                                                         algorithm.args=list(score=used_score, tabu = tabu_para, max.tabu = tabu_para, max.iter = tabu_para)))
  })

cri_bds_ta100_10_d1 = direct_net(cri_bds_ta100_10_d1, used_score, Data10_2_d1)

bayes_cri_bds_ta100_10_d1 = compute_prob_v2(cri_bds_ta100_10_d1, 'RUSSIA', Data10_2_d1)

print('bds tabu 100 cri')
print(as.list(tm_bds_ta100_cri)$elapsed)

if (algorithm == 'tabu'){
  alg = paste(algorithm, tabu_para, sep = '_')
}else{
  alg = algorithm
}

time_matrix[paste(used_score, period, sep = '_'), alg] = as.list(tm_bds_ta100_cri)$elapsed

filename_10 = "C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Results/Networks/networks_and_data_10_d1.RData"

save(time_matrix, pre_k2_hc_10_d1, pre_k2_ta50_10_d1, pre_k2_ta100_10_d1, pre_bds_hc_10_d1, pre_bds_ta50_10_d1, pre_bds_ta100_10_d1,
     cri_k2_hc_10_d1, cri_k2_ta50_10_d1, cri_k2_ta100_10_d1, cri_bds_hc_10_d1, cri_bds_ta50_10_d1, cri_bds_ta100_10_d1,
     Data10_1_d1, Data10_2_d1, file = filename_10)

filename_prob_10 = "C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Results/Networks/probab_10_d1.RData"

save(bayes_pre_k2_hc_10_d1, bayes_pre_k2_ta50_10_d1, bayes_pre_k2_ta100_10_d1, bayes_pre_bds_hc_10_d1,
     bayes_pre_bds_ta50_10_d1, bayes_pre_bds_ta100_10_d1, bayes_cri_k2_hc_10_d1, bayes_cri_k2_ta50_10_d1,
     bayes_cri_k2_ta100_10_d1, bayes_cri_bds_hc_10_d1, bayes_cri_bds_ta50_10_d1, bayes_cri_bds_ta100_10_d1,
     file = filename_prob_10)

## END COPY