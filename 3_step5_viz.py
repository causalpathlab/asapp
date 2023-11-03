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



import os
import fnmatch

def find_files_with_eval(root_dir):
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*eval.csv'):
            matches.append(os.path.join(root, filename))
    return matches


def plot_eval(dfm,method):
    df = dfm[dfm['method']==method]
    grps = ['method','model','size','n_topics','res']
    df = df.groupby(grps).agg(['mean','std' ]).reset_index()
    df.columns = df.columns.map('_'.join).str.strip('_')
    df = df.drop(columns=['seed_mean','seed_std'])
    # df = df.dropna()
    # df = df.fillna(0.01)

    # df = df[df['model'] !="base"]

    for x,y in pairs:
        p = (
            ggplot(df, aes(x=x,y='score_mean',color='model')) +
            geom_pointrange(data=df, mapping=aes(x=x, ymin='score_mean - score_std', ymax='score_mean + score_std'),linetype='solid',size=0.8) +
            scale_color_manual(values=custom_palette) +
            geom_line(data=df, mapping=aes(x=x, y='score_mean', color='model'), linetype='solid',size=0.8) + 
            facet_wrap('~'+y) +
            labs(x=x, y=method)
        )
        p = p + theme(
                plot_background=element_rect(fill='white'),
                panel_background = element_rect(fill='white')
        )
        p.save(filename = wdir+'nmf_eval_'+method+'_'+x+'_'+y+'.png', height=6, width=8, units ='in', dpi=300)


pairs = [
    ('res','size')
]


wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/examples/pbmc/results/'
flist = find_files_with_eval(wdir)

df = pd.DataFrame()
for f in flist:
    print(f)
    dfc = pd.read_csv(f)
    dfc['sample'] = os.path.basename(f).replace('_eval.csv','')
    df = pd.concat([df,dfc],axis=0,ignore_index=True)

df['sample'] = [x.split('_')[0] for x in df['sample']]
print(df['sample'].value_counts())
print(df)

df = df.loc[df['model'] != 'base',:]

plot_eval(df,'Purity')
plot_eval(df,'ARI')
plot_eval(df,'NMI')
