library(xlsx)
library(dplyr)
library(bnlearn)

setwd('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/CDS_data/toy_data')
data = read.xlsx2('toydata.xlsx', sheetIndex = 1)

data = as.data.frame(t(data))

bic_ned = tabu(data, score = 'bic', tabu = 10, max.tabu = 10)

k2_ned = tabu(data, score = 'k2', tabu = 10, max.tabu = 10)

bic_fitted = bn.fit(bic_ned, data)
k2_fitted = bn.fit(k2_ned, data)

data_v = as.data.frame(t(read.xlsx2('toy_vstructure.xlsx', sheetIndex = 1)))

v_bic_net = tabu(data_v, score = 'bic', tabu = 10, max.tabu = 10)

v_bic_fit = bn.fit(v_bic_net, data_v)