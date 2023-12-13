import pandas as pd
import numpy as np
import glob
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc
from plotnine import *
import os
import sys 

# results = sys.argv[1]
# wdir = os.getcwd()+'/'+results+'/'

wdir = 'experiments/asapp/figures/fig_3_c/25k_results/'

custom_palette = ["#d62728","#385E0F" ]

flist = []
for name in glob.glob(wdir+'*_all_result.csv'):
    flist.append(name)

df = pd.DataFrame()
for f in flist:
    dfc = pd.read_csv(f)
    df = pd.concat([df,dfc],axis=0,ignore_index=True)

def plot_eval(dfm,method):
    df = dfm[dfm['method']==method]
    grps = ['method','model','rho','depth','size','topic','res']
    df = df.groupby(grps).agg(['mean','std' ]).reset_index()
    df.columns = df.columns.map('_'.join).str.strip('_')
    df = df.drop(columns=['seed_mean','seed_std'])


    df['size'] = df['size'] * 13

    if method in ['ARI','NMI','Purity']:
        yl = 0
        yh = 1
    elif method in ['Memory']:
        yl = 0
        yh = df['score_mean'].max()+2
    else:
        yl = 0
        yh = df['score_mean'].max()+5

    x= 'size'
    p = (
        ggplot(df, aes(x=x,y='score_mean',color='model')) +
        geom_pointrange(data=df, mapping=aes(x=x, ymin='score_mean - score_std', ymax='score_mean + score_std'),linetype='solid',size=0.5) +
        scale_color_manual(values=custom_palette) +
        geom_line(data=df, mapping=aes(x=x, y='score_mean', color='model'), linetype='solid',size=2.0) + 
        # facet_wrap('~'+y) +
        labs(x=x, y=method)+
        ylim(yl, yh)
    )
    p = p + theme(
            plot_background=element_rect(fill='white'),
            panel_background = element_rect(fill='white')
    )
    p.save(filename = wdir+'nmf_eval_'+method+'_'+x+'_.pdf', height=6, width=8, units ='in',dpi=600)


df = df.loc[df['model']!='scanpy']
df = df.loc[df['model']!='asapf']
df = df.loc[df['model']!='nmf']
plot_eval(df,'Purity')
plot_eval(df,'ARI')
plot_eval(df,'NMI')


df.loc[df['method']=='Memory',['score']] = df.loc[df['method']=='Memory',['score']] * 0.001048576 
df.loc[df['method']=='Time',['score']] = df.loc[df['method']=='Time',['score']] / 60

plot_eval(df,'Memory')
plot_eval(df,'Time')
