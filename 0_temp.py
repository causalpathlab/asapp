import pandas as pd
import numpy as np
import glob
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc
from plotnine import *

f='BRCA_GSE176078_CellMetainfo_table.tsv'

df = pd.read_csv(f,sep='\t')
df = df[['Cell','Celltype (major-lineage)']]
df.columns = ['cell','celltype']
df.celltype.value_counts()
df.to_csv('brca2_celltype.csv.gz',index=False,compression='gzip')