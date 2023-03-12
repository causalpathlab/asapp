library(ggplot2)
library(gridExtra)
library(reshape)
library(pheatmap)
library(RColorBrewer)
library(Polychrome)

struct_plot <- function(df_h,f) {

df_h_m = melt(df_h,id=c("cell","topic"))
df_h_m$topic <- factor(df_h_m$topic)

colnames(df_h_m) = c("cell", "cluster", "topic", "hvalue")

df_h_m$topic <- gsub("X","",df_h_m$topic)
# df_h_m$Topic <- factor(df_h_m$Topic, levels = c("0", "1", "2","3","4","5","6","7","8","9","10","11","12","13","14"))

n <- 10
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))
# col_vector = c("#F0A3FF", "#0075DC","#808080" ,"#4C005C","#2BCE48","#FFCC99","#993F00","#94FFB5","#8F7C00","#9DCC00","#C20088","#003380","#FFA405","#FFA8BB","#426600","#FF0010","#5EF1F2","#00998F","#740AFF","#990000","#FFFF00")

# col_vector <- distinctColorPalette(25)

p <-
ggplot(df_h_m, aes(x=cell, y=hvalue,fill=topic)) +
  geom_bar(position="stack",stat="identity",size=0) +
  scale_fill_manual("ETM topic",values=col_vector)+
  facet_grid(~ cluster, scales = "free", switch = "x", space = "free")+
  labs(x = "Cells", y = "Topic proportion")+
  theme(
    legend.position = "top",
    legend.justification = "left", 
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin=margin(10,10,10,10),
    text = element_text(size=25),
    panel.spacing.x = unit(0.005, "lines"),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank(),
    panel.background = element_rect(fill='transparent'),
    plot.background = element_rect(fill='transparent', color=NA))+
  guides(fill = guide_legend(nrow = 1))

ggsave(f,p,width = 60, height = 10,limitsize=F)
}
