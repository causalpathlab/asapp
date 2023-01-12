library(ggplot2)
library(gridExtra)
library(reshape)
library(yaml)
library(RColorBrewer)
library(Polychrome)
library(data.table)
library(ggh4x)
source("Util.R")


summary_plot_v1 <- function(df,f) {

  col_vector <- c("orange", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941","yellowgreen" , "#7A4900","#FFDBE5",  "#0000A6")

  df$topic = as.factor(df$topic)
 p = ggplot(df, aes(fill=cell_type, y=ncount, x=topic)) + scale_fill_manual("Cell type ",values=col_vector)+
    geom_bar(position="stack", stat="identity") +
      labs(x = "Topics", y = "Celltype distribution")+
  theme(
    legend.position = "right",
    legend.justification = "left", 
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin=margin(10,10,10,10),
    text = element_text(size=75),
    panel.spacing.x = unit(0.005, "lines"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank())
    # panel.background = element_rect(fill='transparent'),
    # plot.background = element_rect(fill='transparent', color=NA))+
  # guides(fill = guide_legend(nrow = n))

ggsave(f,p,width = 30, height = 20,limitsize=F)
}

summary_boxplot<- function(df_summary,f){
# grouped boxplot
col_vector <- as.vector(kelly.colors(22))[16:22]
df_summary$interact_topic = as.factor(df_summary$interact_topic)
p <- ggplot(df_summary, aes(x=interact_topic, y=ncount, fill=interact_topic)) + 
    geom_boxplot()+
    facet_wrap(~celltype,ncol=9)+
    labs(x ="", y = "")+
    scale_fill_manual("Interaction topic",values=col_vector)+    
      theme(
      legend.position = "none",
      text = element_text(size=25),
      panel.spacing.x = unit(0.1, "lines"),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid = element_blank(),
      panel.background = element_rect(fill='grey95'),
      plot.background = element_rect(fill='transparent', color=NA))+
    guides(fill = guide_legend(nrow = 1))

ggsave(f,p,width =10, height = 4)
}