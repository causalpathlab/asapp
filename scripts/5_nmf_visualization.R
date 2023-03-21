
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
df_eval = read.table(paste(paste(args$home,args$experiment,args$output,args$sample_id,'/',sep=''),"eval_result.csv",sep=""), sep = ",", header=TRUE)

p <- ggplot(df_eval, aes(x=rho, y=score_mean, group = mode, color=mode))+ 
    geom_errorbar(aes(ymin=score_mean-score_std, ymax=score_mean+score_std), width=.1, position=position_dodge(0.05)) +
    geom_line(aes(linetype=mode)) + 
    geom_point(aes(shape=mode))+
    facet_wrap(~size)+
    labs(title=" noise (1 - rho) and data size (celltype)",x="rho", y = "NMI")+
    theme_classic() + scale_color_manual(values=c('#E69F00',"#AE4371",'#999999'))

ggsave(paste(paste(args$home,args$experiment,args$output,args$sample_id,'/',sep=''),"eval_result.pdf",sep=""),p,width = 10, height = 3,limitsize=F)

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
