library(ggplot2)
library(gridExtra)
library(reshape)
library(yaml)
library(pheatmap)
library(viridis)
library(dplyr)
source("Util.R")
# options(repr.plot.width = 15, repr.plot.height = 7, repr.plot.res = 300)


beta_loadings_plot <- function(df_beta,df_tg,f) {

top_genes = unique(df_tg$Gene)
df_beta = select(df_beta,all_of(top_genes))

row_order = row.order(df_beta)

df_beta_t = df_beta
df_beta_t$topic = rownames(df_beta)
df_beta_t = melt(df_beta_t)
colnames(df_beta_t)=c('row','col','weight')
col_order = col.order(df_beta_t,row_order)

df_beta = df_beta[,col_order]
df_beta = df_beta[row_order,]


p1 <- pheatmap(t(df_beta),fontsize_row=8,fontsize_col=8,cluster_rows=FALSE,cluster_cols=FALSE,show_colnames=,show_rownames=FALSE)

ggsave(f,p1,dpi = 300)
}
