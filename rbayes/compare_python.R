library(bnlearn)
library(Rgraphviz)
library(xlsx)

plot_net <- function(bayesian_net){
  g1 = as.graphAM(bayesian_net)
  plot(g1, attrs=list(node=list(fillcolor="lightgreen", fontsize = 15),edge=list(color="black")))
}

general_path = 'C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/CDS_data/toy_data'

files = list('toydata.xlsx','toy_vstructure.xlsx')

toy = as.data.frame(t(read.xlsx(file.path(general_path, files[1]), sheetIndex = 1)))
toy_v = as.data.frame(t(read.xlsx(file.path(general_path, files[2]), sheetIndex = 1)))

rownames(toy) = as.character(unlist(1:nrow(toy)))
rownames(toy_v) = as.character(unlist(1:nrow(toy_v)))

colnames(toy) = as.character(unlist(toy[1,]))
toy = toy[-1,]
colnames(toy_v) = as.character(unlist(toy_v[1,]))
toy_v = toy_v[-1,]

toy <- as.data.frame(lapply(toy, function(x)
    if(is.factor(x)) factor(x) else x))

toy_v <- as.data.frame(lapply(toy_v, function(x)
  if(is.factor(x)) factor(x) else x))

used_score = 'bic'
RR = 1000
algorithm = 'hc'

net = averaged.network(boot.strength(data = toy, R = RR, algorithm = algorithm,
                                     algorithm.args=list(score=used_score)))

net_v = averaged.network(boot.strength(data = toy_v, R = RR, algorithm = algorithm,
                                     algorithm.args=list(score=used_score)))
plot_net(net_v)