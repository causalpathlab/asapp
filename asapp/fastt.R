library(Matrix)
library(fastTopics)
library(ggplot2)
library(cowplot)
set.seed(1)

fpath = '/home/BCCRC.CA/ssubedi/projects/experiments/fastsca/result/tnbc/tnbc_rp'

f = '/home/BCCRC.CA/ssubedi/projects/experiments/fastsca/result/tnbc/tnbc_rp_bulk.csv.gz'
df = read.table(f,sep=',',header=TRUE)
fit <- fit_topic_model(as.matrix(df),k =10)

