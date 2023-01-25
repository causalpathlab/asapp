################################ set up ##################
args_home ="/home/BCCRC.CA/ssubedi/projects/experiments/asapp/"
setwd(paste(args_home,'asapp/r_scripts/',sep=''))
library(yaml)
options(rlib_downstream_check = FALSE)
config = paste(args_home,"config.yaml",sep="") 
args = read_yaml(config)
model_id = paste(args_home,args$output,args$sample_id,'/',args$sample_id,sep='')


###############################################################################
# plot analysis 
###############################################################################

source('fig_topic_ct_distribution.R')
summary_file = paste(model_id,'_r1_topic_ct_dist.csv.gz',sep="")
df = read.table(summary_file,sep=',', header=TRUE)
f = paste(model_id,'_r1_topic_ct_dist.pdf',sep="")
summary_plot_v1(df,f)    

# summary_file = paste(model_id,'_r1_topic_bulk_ct_dist.csv.gz',sep="")
# df = read.table(summary_file,sep=',', header=TRUE)
# f = paste(model_id,'_r1_topic_bulk_ct_dist.pdf',sep="")
# summary_plot_v1(df,f)    
