library(umap)
df = read.table('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/simdata/simdata_theta.csv.gz',sep=',',row.names=1,header=TRUE)
head(df)
dfumap=umap(df)
library(readr)
write.csv(dfumap$layout,'/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/simdata/simdata_theta_umap.csv')

