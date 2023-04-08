
################################ set up ##################
library(yaml)


args = read_yaml('config.yaml')
model_id = paste(args$home,args$experiment,args$output,args$sample_id,'/',args$sample_id,sep='')
data_path = paste(args$home,args$experiment,args$input,args$sample_id,'/',args$sample_id,sep='')

print(model_id)
###############################################################################
# model  
###############################################################################




###############################################################################

library(ggplot2)
library(RColorBrewer)
library(Polychrome)
library(gridExtra)

df_eval = read.table(paste(paste(args$home,args$experiment,args$output,args$sample_id,'/',sep=''),"eval_result.csv",sep=""), sep = ",", header=TRUE)

col_vector = as.vector(kelly.colors(22)[3:10])


plotlist = list()
sel_vector = unique(df_eval$size) 
for (i in 1:2){
t = paste("rho,size=",sel_vector[i],sep='')
p <- ggplot(df_eval[df_eval$size == sel_vector[i],], aes(x=rho, y=score_mean, group = mode, color=mode))+ 
    geom_errorbar(aes(ymin=score_mean-score_std, ymax=score_mean+score_std), width=.1, position=position_dodge(0.01)) +
    geom_line(linewidth=0.7) + 
    # facet_wrap(~alpha,ncol=5)+
    labs(title=t,x="rho", y = "NMI")+
    theme_classic() + scale_color_manual(values=col_vector)
plotlist[[i]] = p
}
stplt <- grid.arrange(grobs=plotlist,nrow=2,
heights = c(1/2,1/2))
ggsave(paste(paste(args$home,args$experiment,args$output,args$sample_id,'/',sep=''),"eval_result_depth.pdf",sep=""),stplt,width = 10, height = 15,limitsize=F)

# plotlist = list()
# sel_vector = unique(df_eval$size) 
# for (i in 1:5){
# t = paste("rho,size=",sel_vector[i],sep='')
# p <- ggplot(df_eval[df_eval$depth == sel_vector[i],], aes(x=rho, y=score_mean, group = mode, color=mode))+ 
#     geom_errorbar(aes(ymin=score_mean-score_std, ymax=score_mean+score_std), width=.1, position=position_dodge(0.01)) +
#     geom_line(linewidth=0.7) + 
#     # facet_wrap(~alpha,ncol=5)+
#     labs(title=t,x="rho", y = "NMI")+
#     theme_classic() + scale_color_manual(values=col_vector)
# plotlist[[i]] = p
# }
# stplt <- grid.arrange(grobs=plotlist,nrow=5,
# heights = c(1/5, 1/5, 1/5,1/5,1/5))
# ggsave(paste(paste(args$home,args$experiment,args$output,args$sample_id,'/',sep=''),"eval_result_depth.pdf",sep=""),stplt,width = 10, height = 15,limitsize=F)

#  #############
# library(reticulate)
# np = import('numpy')
# model = np$load(paste(model_id,'_dcnmf.npz',sep=''))

# ###############################################################################
# # cell topic analysis 
# ###############################################################################

# ### cell topic heatmap plot 
# source('fig_beta_hmap.R')
# df_tg = read.table(paste(model_id,"_beta_top_genes.csv.gz",sep=""), sep = ",", header=TRUE)

# df_beta = as.data.frame(t(exp(model$f['beta'])))

# beta_genes = read.csv(paste(data_path,".cols.csv.gz",sep=""),header=TRUE)
# colnames(df_beta)=beta_genes$cols
# rownames(df_beta)=0:9

# f = paste(model_id,"_result_beta_top_genes_hmap.pdf",sep="")
# beta_loadings_plot(df_beta,df_tg,f)




# ### cell topic structure plot
# source('fig_struct_plot.R')
# h_sample_file = paste(model_id,"_theta_sample_topic.csv.gz",sep="")
# df_h = read.table(h_sample_file, sep = ",", header=TRUE)
# struct_plot(df_h,paste(model_id,'_theta_sample_topic.pdf',sep=''))
