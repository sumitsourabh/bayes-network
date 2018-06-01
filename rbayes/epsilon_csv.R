library(bnlearn)
library(readr)
library(Rgraphviz)
library(tictoc)

tic("BN")

setwd('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/CDS_data/epsilon_drawups')

source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')


ep_draw = reading(10,3)

# ep_draw = read_test()

used_score = "bic"

# tic("boot algorithm tabu")
tabu_param = 30
boot = boot.strength(data = ep_draw, R = 200, algorithm="tabu", algorithm.args=list(score=used_score, tabu = tabu_param, max.tabu = tabu_param, max.iter = tabu_param))
# toc()

# tic("boot algorithm hc")
# boot = boot.strength(data = ep_draw, R = 100, algorithm="hc", algorithm.args=list(score=used_score))
# toc()

avg_net = averaged.network(boot)

plot_net(avg_net)

final_net = direct_net(avg_net, used_score, ep_draw)

param.bayes = bn.fit(final_net, data = ep_draw, method = 'bayes')
CPGSD_bayes = compute_prob(nodes(final_net), 'Russian.Fedn', param.bayes)
# print(CPGSD_bayes)
# CPGSD_inv_bayes = compute_inv_prob(nodes(final_net), 'Russian.Fedn', param.bayes)
# print(CPGSD_inv_bayes)

param.mle = bn.fit(final_net, data = ep_draw, method = 'mle')
CPGSD_mle = compute_prob(nodes(final_net), 'Russian.Fedn', param.mle)
# print(CPGSD_mle)
# CPGSD_inv_mle = compute_inv_prob(nodes(final_net), 'Russian.Fedn', param.mle)
# print(CPGSD_inv_mle)

toc()
