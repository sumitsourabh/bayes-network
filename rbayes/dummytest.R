library(bnlearn)
library(Rgraphviz)

source('C:/Users/Javier/Documents/MEGA/Universitattt/Master/Thesis/Code/rbayes/functions.R')

df = data.frame(matrix(0.0,5,3))
colnames(df) = c('A','B','C')
df[1,'A'] = 1.0
df[1,'B'] = 0.5
df[4,'B'] = 1.0
df[4,'C'] = 0.5
df[5,'C'] = 1.0

df = as.data.frame(lapply(df, function(x) if(is.factor(x)) x else as.factor(x)))

used_score = "bds"

tic("boot algorithm hc")
boot = boot.strength(data = df, R = 200, algorithm="hc", algorithm.args=list(score=used_score))
toc()

avg_net = averaged.network(boot)

plot_net(avg_net)

final_net = direct_net(avg_net, used_score)

paramet = bn.fit(final_net, data = df, method = 'bayes')
cp = data.frame(c(0.0,0.0), row.names = c('B','C'))
colnames(cp) = c('CP')
cp['B','CP'] = cpquery(paramet, event = (B == 0.5), evidence = (A == 1))
cp['C','CP'] = cpquery(paramet, event = (C == 0.5), evidence = (A == 1))
print(cp)