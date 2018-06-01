library(bnlearn)
library(readr)
library(Rgraphviz)
library(tictoc)

tm1 <- system.time(
  {
    tic("total")
    
    source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')
    
    Data = reading(10,3)
    
    used_score = 'k2'
    R = 200
    algorithm = 'tabu'
    # tabu parameters (if needed)
    tabu_param = 13
    
    # MonteCarlo iterations
    iter_mont = 100
    
    counterparties = c("Gazprom.Neft","Mobile.Te.Sy","Transneft","City.Moscow",
                       "VIMPELCOM","SBERBANK","Ru.Railways","Ru.Agric.Bk",
                       "ALROSA","Rosneft","Vnesheconbk","Bank.Moscow",
                       "GAZPROM","VTB.Bk","MDM.Bk","Ru.Std.Bk",
                       "Evraz.Group","Lukoil")
    
    tot_bayes = as.data.frame(matrix(0.0,18,iter_mont), row.names = counterparties)
    colnames(tot_bayes) = 1:iter_mont
    tot_mle = as.data.frame(matrix(0.0,18,iter_mont), row.names = counterparties)
    colnames(tot_mle) = 1:iter_mont
    
    adjmat = as.data.frame(matrix(0, 19, 19), row.names = append(counterparties,'Russian.Fedn'))
    colnames(adjmat) = row.names(adjmat)
    
    for (i in 1:iter_mont){
      print(i)
      tic('BN')
      # source('~/MEGA/Universitattt/Master/Thesis/Code/rbayes/epsilon_csv.R')
      unpack[CPGSD_bayes, CPGSD_mle, final_net] = BN(Data = Data, used_score = used_score, RR = R, algthm = algorithm, parameter = tabu_param)
      for (c in counterparties){
        tot_mle[c,i] = CPGSD_mle[c,]
        tot_bayes[c,i] = CPGSD_bayes[c,]
      }
      arcs_net = arcs(final_net)
      for (i in 1:nrow(arcs_net)){
        adjmat[arcs_net[i,1],arcs_net[i,2]] = adjmat[arcs_net[i,1],arcs_net[i,2]] + 1
      }
      toc()
    }
    
    # Mean and variance of probabilities:
    # Bayes method
    prob_bayes = as.data.frame(matrix(0.0,18,2),row.names = counterparties)
    colnames(prob_bayes) = c('mean','std')
    prob_bayes[,'mean'] = apply(tot_bayes,1,mean)
    prob_bayes[,'std'] = apply(tot_bayes,1,sd)
    prob_bayes = prob_bayes[order(-prob_bayes$mean), , drop = FALSE]
    
    if (algorithm == 'tabu'){
      alg = paste(algorithm, as.character(tabu_param), sep = '')
    } else{
      alg = algorithm
    }
    
    main_directory_name = 'C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Results/Probabilities'
    
    sub_directory = paste(used_score, alg, R, iter_mont, sep = '_')
    
    dir.create(file.path(main_directory_name, sub_directory), showWarnings = FALSE)
    
    filename_bayes = 'bayes_prob.csv'
    
    write.csv(prob_bayes, file.path(main_directory_name, sub_directory, filename_bayes),
              append = FALSE, row.names = TRUE, col.names = TRUE)
    
    # MLE method
    prob_mle = as.data.frame(matrix(0.0,18,2),row.names = counterparties)
    colnames(prob_mle) = c('mean','std')
    prob_mle[,'mean'] = apply(tot_mle,1,mean)
    prob_mle[,'std'] = apply(tot_mle,1,sd)
    prob_mle = prob_mle[order(-prob_mle$mean), , drop = FALSE]
    
    filename_mle = 'mle_prob.csv'
    
    write.csv(prob_mle, file.path(main_directory_name, sub_directory, filename_mle),
              append = FALSE, row.names = TRUE, col.names = TRUE)
    
    adjmat = adjmat/iter_mont
    
    filename_matrix = 'adj_matrix.csv'
    
    write.csv(adjmat, file.path(main_directory_name, sub_directory, filename_matrix),
              append = FALSE, row.names = TRUE, col.names = TRUE)
    
    
    toc()
  })

t = as.list(tm1)$elapsed

write(t, file = file.path(main_directory_name, sub_directory, 'time_elapsed.txt'))