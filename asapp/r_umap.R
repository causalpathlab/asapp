library(umap)
df = read.table('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/hbcc/hbcc_sc_theta.csv.gz',sep=',')
dim(df)
df = read.table('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/hbcc/hbcc_sc_theta.csv.gz',sep=',',row.names=1)
head(df)
df = read.table('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/hbcc/hbcc_sc_theta.csv.gz',sep=',',row.names=1,header=True)
df = read.table('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/hbcc/hbcc_sc_theta.csv.gz',sep=',',row.names=1,header=TRUE)
head(df)
dfumap=umap(df)
write.csv.gz(dfumap$layout,'/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/hbcc/hbcc_sc_umap.csv.gz')
library(readr)
write.csv(dfumap$layout,'/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/hbcc/hbcc_sc_umap.csv.gz')
dfumap$layout
q
history
history()
exit()
