library(xlsx)
library(dplyr)
library(bnlearn)

setwd('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/CDS_data/toy_data')
data_n = as.data.frame(t(read.xlsx2('toy_normal.xlsx', sheetIndex = 1)))
data_normal = data.frame(V1 = as.numeric(as.character(data_n$V1)), V2 = as.numeric(as.character(data_n$V2)), V3 = as.numeric(as.character(data_n$V3)))

data_v = as.data.frame(t(read.xlsx2('toy_vstruc_normal.xlsx', sheetIndex = 1)))
data_v_struc = data.frame(V1 = as.numeric(as.character(data_v$V1)), V2 = as.numeric(as.character(data_v$V2)), V3 = as.numeric(as.character(data_v$V3)))

boot = boot.strength(data = data_v_struc, R = 100, algorithm="tabu", algorithm.args=list(score="bic-g"))
avg.boot = averaged.network(boot)
avg.fitted = bn.fit(avg.boot, data = data_v_struc)

bic_ned_normal = tabu(data_normal, score = 'bic-g', tabu = 10, max.tabu = 10)
bic_fitted_normal = bn.fit(bic_ned_normal, data_normal)

boot = boot.strength(data = data_normal, R = 100, algorithm="tabu", algorithm.args=list(score="bic-g"), cpdag = FALSE)
avg.boot = averaged.network(boot)
avg.fitted = bn.fit(avg.boot, data = data_normal)

v_bic_net_normal = tabu(data_v_struc, score = 'bic-g', tabu = 10, max.tabu = 10)
v_bic_fit_normal = bn.fit(v_bic_net_normal, data_v_struc)
