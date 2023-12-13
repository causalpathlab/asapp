import pandas as pd
import numpy as np
import glob
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc
from plotnine import *
import os
import sys 

result = sys.argv[1]
wdir = os.getcwd()+'/'+result+'/'

custom_palette = ["#7f7f7f",
"#1f77b4", "#2ca02c","#bcbd22", "#9467bd",
"#8c564b", "#e377c2" ]

flist = []
for name in glob.glob(wdir+'*_eval.csv'):
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
    # df = df.dropna()
    # df = df.fillna(0.01)


    df['size'] = df['size'] * 13

    # df = df[df['model'] !="base"]

    for x,y in pairs:
        p = (
            ggplot(df, aes(x=x,y='score_mean',color='model')) +
            geom_pointrange(data=df, mapping=aes(x=x, ymin='score_mean - score_std', ymax='score_mean + score_std'),linetype='solid',size=0.2) +
            scale_color_manual(values=custom_palette) +
            geom_line(data=df, mapping=aes(x=x, y='score_mean', color='model'), linetype='solid',size=0.4) + 
            facet_wrap('~'+y) +
            labs(x=x, y=method)
        )
        p = p + theme(
                plot_background=element_rect(fill='white'),
                panel_background = element_rect(fill='white')
        )
        p.save(filename = wdir+'nmf_eval_'+method+'_'+x+'_'+y+'.pdf', height=6, width=8, units ='in', dpi=600)


pairs = [
    # ('rho','size'),
    # ('rho','depth'),
    ('rho','res'),
    # ('res','rho'),
    # ('rho','topic'),
]

def combine_plot(dfm,method):
    df = dfm[dfm['method']==method].copy()
    grps = ['method','model','rho','depth','size','topic','res']
    df = df.groupby(grps).agg(['mean','std' ]).reset_index()

    df.columns = df.columns.map('_'.join).str.strip('_')

    df = df.drop(columns=['seed_mean','seed_std'])
    grps = ['method','model','rho','depth','size','topic']
    df2 = df.groupby(grps).agg(['mean','std' ]).reset_index()

    df2.columns = df2.columns.map('_'.join).str.strip('_')
    df2 = df2.drop(columns=['res_mean','res_std','score_mean_std','score_std_std'])

    p = (
        ggplot(df2, aes(x='rho',y='score_mean_mean',color='model')) +
        geom_pointrange(data=df2, mapping=aes(x='rho', ymin='score_mean_mean - score_std_mean', ymax='score_mean_mean + score_std_mean'),linetype='solid',size=0.5) +
        scale_color_manual(values=custom_palette) +
        geom_line(data=df2, mapping=aes(x='rho', y='score_mean_mean', color='model'),size=2.0) +
        labs(x='rho', y=method)

    )
    p = p + theme(
            plot_background=element_rect(fill='white'),
            panel_background = element_rect(fill='white')
    )
    p.save(filename = wdir+'nmf_eval_comb_'+method+'.pdf', height=6, width=8, units ='in', dpi=600)


df = df[((df['depth']==10000) & (df['size']==250) & (df['topic']==13))]

plot_eval(df,'Purity')
plot_eval(df,'ARI')
plot_eval(df,'NMI')

combine_plot(df,method='NMI')
combine_plot(df,method='Purity')
combine_plot(df,method='ARI')