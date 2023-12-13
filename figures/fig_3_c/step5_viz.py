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

custom_palette = [
"#c40233" ,"#800020","#f08080" ]


wdir = 'experiments/asapp/figures/fig_3_c/25k_results/'
flist = []
for name in glob.glob(wdir+'*_all_result.csv'):
    flist.append(name)

df1 = pd.DataFrame()
for f in flist:
    dfc = pd.read_csv(f)
    df1 = pd.concat([df1,dfc],axis=0,ignore_index=True)

wdir = 'experiments/asapp/figures/fig_3_c/100k_results/'
flist = []
for name in glob.glob(wdir+'*_all_result.csv'):
    flist.append(name)

df2 = pd.DataFrame()
for f in flist:
    dfc = pd.read_csv(f)
    df2 = pd.concat([df2,dfc],axis=0,ignore_index=True)

wdir = 'experiments/asapp/figures/fig_3_c/200k_results/'
flist = []
for name in glob.glob(wdir+'*_all_result.csv'):
    flist.append(name)

df3 = pd.DataFrame()
for f in flist:
    dfc = pd.read_csv(f)
    df3 = pd.concat([df3,dfc],axis=0,ignore_index=True)


df1['model'] = [x+'_25k' for x in df1['model']]
df2['model'] = [x+'_100k' for x in df2['model']]
df3['model'] = [x+'_200k' for x in df3['model']]
df = pd.concat([df1,df2,df3],axis=0,ignore_index=True)

wdir = 'experiments/asapp/figures/fig_3_c/'

def plot_eval(dfm,method):
    df = dfm[dfm['method']==method]
    grps = ['method','model','rho','depth','size','topic','res']
    df = df.groupby(grps).agg(['mean','std' ]).reset_index()
    df.columns = df.columns.map('_'.join).str.strip('_')
    df = df.drop(columns=['seed_mean','seed_std'])
    # df = df.dropna()
    # df = df.fillna(0.01)


    df['size'] = df['size'] * 13

    if method in ['ARI','NMI','Purity']:
        yl = 0
        yh = 1
    elif method in ['Memory']:
        yl = 0
        yh = df['score_mean'].max()+2.5
    
    elif method in ['Time']:
        yl = 0
        yh = df['score_mean'].max()+1
    elif method in ['pbtotal']:
        yl = 0
        yh = df['score_mean'].max()+2500
        
    elif method in ['pbsize']:
        yl = 0
        yh = df['score_mean'].max()+1000
        
        
    x= 'size'
    p = (
        ggplot(df, aes(x=x,y='score_mean',color='model')) +
        # geom_pointrange(data=df, mapping=aes(x=x, ymin='score_mean - score_std', ymax='score_mean + score_std'),linetype='solid',size=0.5) +
        geom_line(data=df, mapping=aes(x=x, y='score_mean', color='model'), linetype='solid',size=2) + 
        scale_color_manual(values=custom_palette)+
        geom_ribbon(aes(ymin='score_mean - score_std', ymax='score_mean + score_std', fill='model'),alpha=0.2,outline_type='upper') +
        # facet_wrap('~'+y) +
        scale_fill_manual(values=custom_palette) +
        labs(x=x, y=method)+
        ylim(yl, yh)
    )
    p = p + theme(
            plot_background=element_rect(fill='white'),
            panel_background = element_rect(fill='white')
    )
    p.save(filename = wdir+'nmf_eval_'+method+'_'+x+'_.png', height=6, width=8, units ='in',dpi=600)



# df = df[((df['depth']==10000) & (df['size']==250) & (df['topic']==13))]
df = df.loc[df['model']!='scanpy']
df = df.loc[df['model']!='asapf']
df = df.loc[df['model']!='nmf']
df = df.loc[df['model']!='liger']


df.loc[df['method']=='Memory',['score']] = df.loc[df['method']=='Memory',['score']] * 0.001048576 
df.loc[df['method']=='Time',['score']] = df.loc[df['method']=='Time',['score']] / 60


plot_eval(df,'Purity')
plot_eval(df,'ARI')
plot_eval(df,'NMI')
plot_eval(df,'Memory')
plot_eval(df,'Time')
plot_eval(df,'pbsize')
plot_eval(df,'pbtotal')
