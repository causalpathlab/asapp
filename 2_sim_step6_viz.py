import pandas as pd
import numpy as np
import glob
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc
from plotnine import *


custom_palette = [
"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
"#8c564b", "#e377c2","#7f7f7f", "#bcbd22", "#17becf"]

flist = []
for name in glob.glob('./results/*_eval.csv'):
    flist.append(name)

df = pd.DataFrame()
for f in flist:
    dfc = pd.read_csv(f)
    df = pd.concat([df,dfc],axis=0,ignore_index=True)

def plot_eval(dfm,method):
    df = dfm[dfm['method']==method]
    grps = ['method','model','rho','size']
    df = df.groupby(grps).agg(['mean','std' ]).reset_index()
    df.columns = df.columns.map('_'.join).str.strip('_')
    df = df.drop(columns=['seed_mean','seed_std'])
    # df = df.dropna()
    df = df.fillna(0.01)


    df['size'] = df['size'] * 13

    # df = df[df['model'] !="base"]


    p = (
        ggplot(df, aes(x='rho',y='score_mean',color='model')) +
        geom_pointrange(data=df, mapping=aes(x='rho', ymin='score_mean - score_std', ymax='score_mean + score_std'),linetype='solid') +
        scale_color_manual(values=custom_palette) +
        geom_line(data=df, mapping=aes(x='rho', y='score_mean', color='model'), linetype='dashed',size=0.5) + 
        facet_wrap('~size') +
        labs(x='rho', y=method)
    )
    p = p + theme(
            plot_background=element_rect(fill='white'),
            panel_background = element_rect(fill='white')
    )
    p.save(filename = './results/nmf_eval_'+method+'.png', height=6, width=8, units ='in', dpi=300)

plot_eval(df,'NMI')
plot_eval(df,'ARI')