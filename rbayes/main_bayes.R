library(bnlearn)
library(readr)
library(Rgraphviz)
library(tictoc)


tm1 <- system.time(
  {
    tic("total")
    
    source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')
    
    wind = 300
    mov = 60
    
    time_para = 10
    delay = 1
    
    percent = 5
    
    date1 = "2012-07-06"
    date2 = "2014-02-26"
    
    used_score = 'k2'
    R = 400
    algorithm = 'hc'
    # tabu parameters (if needed)
    tabu_param = c(1,1,20)
    
    # MonteCarlo iterations
    iter_mont = 10
    
    unpack[Data1, Data2] = read_dates(date1, date2, percent, wind, mov, time_para, delay)
    
    print(paste('Period starting in ', date1, sep = ''))
    
    unpack[bayes1, mle1, mat1, final_net1] <- mont.carl.BN(Data = Data1, score = used_score, bootstrap = R,
                                              algthm = algorithm, param = tabu_param,
                                              iter_mont = iter_mont)
    
    print(paste('Period starting in ', date2, sep = ''))
    
    unpack[bayes2, mle2, mat2, final_net2] <- mont.carl.BN(Data = Data2, score = used_score, bootstrap = R,
                                               algthm = algorithm, param = tabu_param,
                                               iter_mont = iter_mont)
    
    
    main_directory_name = 'C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Results/Probabilities'
    
    if (algorithm == 'tabu'){
      alg = paste(algorithm, as.character(tabu_param[1]),as.character(tabu_param[3]), sep = '-')
    } else{
      alg = algorithm
    }
    
    sub_directory = paste(used_score, alg, R, iter_mont, sep = '_')
    
    dir.create(file.path(main_directory_name, sub_directory), showWarnings = FALSE)
    
    date1_dir = get_folder_date(date1,percent, wind, mov)
    date2_dir = get_folder_date(date2,percent, wind, mov)
    
    dir.create(file.path(main_directory_name, sub_directory, date1_dir), showWarnings = FALSE)
    dir.create(file.path(main_directory_name, sub_directory, date2_dir), showWarnings = FALSE)
    
    filename_bayes1 = paste(date1, 'bayes_prob.csv', sep = '_')
    filename_bayes2 = paste(date2, 'bayes_prob.csv', sep = '_')
    
    write.csv(bayes1, file.path(main_directory_name, sub_directory, date1_dir, filename_bayes1),
              append = FALSE, row.names = TRUE, col.names = TRUE)
    
    write.csv(bayes2, file.path(main_directory_name, sub_directory, date2_dir, filename_bayes2),
              append = FALSE, row.names = TRUE, col.names = TRUE)
    
    filename_mle1 = paste(date1, 'mle_prob.csv', sep = '_')
    filename_mle2 = paste(date2, 'mle_prob.csv', sep = '_')
    
    write.csv(mle1, file.path(main_directory_name, sub_directory, date1_dir, filename_mle1),
              append = FALSE, row.names = TRUE, col.names = TRUE)
    
    write.csv(mle2, file.path(main_directory_name, sub_directory, date2_dir, filename_mle2),
              append = FALSE, row.names = TRUE, col.names = TRUE)
    
    filename_mat1 = paste(date1, 'adj_mat.csv', sep = '_')
    filename_mat2 = paste(date2, 'adj_mat.csv', sep = '_')
    
    write.csv(mat1, file.path(main_directory_name, sub_directory, date1_dir, filename_mat1),
              append = FALSE, row.names = TRUE, col.names = TRUE)
    
    write.csv(mat2, file.path(main_directory_name, sub_directory, date2_dir, filename_mat2),
              append = FALSE, row.names = TRUE, col.names = TRUE)
    
    toc()
  })

t = as.list(tm1)$elapsed

write(t, file = file.path(main_directory_name, sub_directory, 'time_elapsed.txt'))