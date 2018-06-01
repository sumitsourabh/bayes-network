library(xlsx)
library(dplyr)
library(bnlearn)

setwd('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/CDS_data/toy_data')

epsilon_drawups = as.data.frame(t(read.xlsx2('epsilon_drawups.xlsx', sheetIndex = 1)))
# epsilon_drawups = as.data.frame(t(read.xlsx2('epsilon_drawups.xlsx', sheetIndex = 1, colIndex = 2:240)))

bic_ned = tabu(epsilon_drawups, score = 'bic', tabu = 10, max.tabu = 10)
bic_fitted = bn.fit(bic_ned, epsilon_drawups)

boot = boot.strength(data = epsilon_drawups, R = 100, algorithm="hc", algorithm.args=list(score="bic"))
avg.boot = averaged.network(boot)
avg.fitted = bn.fit(avg.boot, data = epsilon_drawups)