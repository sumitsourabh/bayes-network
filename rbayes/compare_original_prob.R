source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')

path = 'C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/CDS_data/Sliced_ep_draw_new/original_dates'

delay = 1

#Define score, algorithm and number of bootstraps

used_score = 'bds'
RR = 1000
algorithm = 'hc'

for (time in list(10,15,20)){
  
  directory1 = paste('std',as.character(time),'long',sep = '_')
  directory2 = paste(directory1, as.character(5),sep = '_')
  
  for (delay in list(1,2,3)){
    tic()
    
    data1 = reading_original(time, delay, path, directory1)
  
    net1 = averaged.network(boot.strength(data = data1, R = RR, algorithm = algorithm,
                                         algorithm.args=list(score=used_score)))
    
    net1 = direct_net(net1, used_score = used_score, Data = data1)
    
    cmd = paste('net_',as.character(time),'_',as.character(delay),' = net1', sep = '')
    eval(parse(text = cmd))
    
    bayes1 = compute_prob_v2(net1,'RUSSIA',data1)
    
    write.csv(bayes1, file = file.path(path, directory1, paste('prob_',as.character(time),'_delay_',
                                                               as.character(delay),'.csv',sep ='')))
    
    data2 = reading_original(time, delay, path, directory2)
    
    net2 = averaged.network(boot.strength(data = data2, R = RR, algorithm = algorithm,
                                          algorithm.args=list(score=used_score)))
    
    net2 = direct_net(net2, used_score = used_score, Data = data2)
    
    cmd = paste('net_',as.character(time),'_',as.character(delay),'_5 = net1', sep = '')
    eval(parse(text = cmd))
    
    bayes2 = compute_prob_v2(net2,'RUSSIA',data2)
    
    write.csv(bayes2, file = file.path(path, directory2, paste('prob_',as.character(time),'_delay_',
                                                               as.character(delay),'_',as.character(5),'.csv', sep = '')))
    cat('Done ')
    cat(as.character(time))
    cat(' with Delay: ')
    cat(as.character(delay))
    cat('\n')
    toc()
  }
}
save(net_10_1,net_10_2,net_10_3,net_15_1,net_15_2,net_15_3,net_20_1,net_20_2,net_20_3, file = file.path(path, 'networks.RData'))

save(net_10_1_5,net_10_2_5,net_10_3_5,net_15_1_5,net_15_2_5,net_15_3_5,net_20_1_5,net_20_2_5,net_20_3_5, file = file.path(path, 'networks_5.RData'))