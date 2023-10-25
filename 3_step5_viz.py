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
        for filename in fnmatch.filter(filenames, '*eval*'):
            matches.append(os.path.join(root, filename))
    return matches


def plot_eval(df):
    df = df.drop(columns=['n_topics','seed','res'])

    p = (
        ggplot(df, aes(x='sample',y='score',color='model')) +
        geom_point()+
        scale_color_manual(values=custom_palette) +
        # geom_line(data=df, mapping=aes(x='sample', y='score', color='model'), linetype='dashed',size=0.5) + 
        facet_wrap('~'+'method') 
    )
    p = p + theme(
            plot_background=element_rect(fill='white'),
            panel_background = element_rect(fill='white')
    )
    p.save(filename = starting_directory+'/nmf_eval_.png', height=6, width=15, units ='in', dpi=300)


starting_directory = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/examples'
flist = find_files_with_eval(starting_directory)

df = pd.DataFrame()
for f in flist:
    print(f)
    dfc = pd.read_csv(f)
    dfc['sample'] = os.path.basename(f).replace('_eval.csv','')
    df = pd.concat([df,dfc],axis=0,ignore_index=True)

df['sample'] = [x.split('_')[0] for x in df['sample']]
df['sample'] = np.where(df['sample']=='brca2','z_brca',df['sample'])
print(df['sample'].value_counts())
plot_eval(df)
# plot_eval(df,'ARI')
# plot_eval(df,'LR')