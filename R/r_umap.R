library(umap)
df = read.table('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/bulkliger_h_norm.csv.gz',sep=',',row.names=1,header=TRUE)
head(df)
dfumap=umap(df,metric='cosine',min_dist=0.5,n_neighbors=25)
library(readr)
write.csv(dfumap$layout,'/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/bulkliger_h_norm_umap.csv')

