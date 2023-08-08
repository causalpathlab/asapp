
from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.factorize import ASAPNMF
from asap.util import analysis
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

from sklearn.preprocessing import StandardScaler
import logging

experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id


logging.basicConfig(filename=sample_out+'_model.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

tree_max_depth = 10
num_factors = 10
batch_size = 25000
downsample_pseudobulk = True
downsample_size = 100
bn = 1

dl = DataSet(sample_in,sample_out)
sample_list = dl.get_dataset_names()
dl.initialize_data(sample_list,batch_size)


dl.dataset_batch_size['bc_332k'] =12500
dl.dataset_batch_size['bc_48k'] =12500
'''
dfm = dl.construct_batch_df(15000)
# batch_label = np.array([x.split('-')[0] for x in dl.barcodes])
batch_label = np.array([x.split('@')[1] for x in dfm.index.values])

import asap.util.plots as pl
pl.plot_stats(dfm,batch_label,dl.outpath+'_pren_stats.png')

import anndata as an
import scanpy as sc
mtx = dfm.to_numpy()
adata = an.AnnData(mtx.T)
sc.pp.normalize_total(adata) 

mtx2 = adata.X.T
dfm2 = pd.DataFrame(mtx2)
dfm2.index = dfm.index
dfm2.columns = dfm.columns

dfm = dfm2

pl.plot_genes_meanvar_barchart(dfm,batch_label,dl.outpath+'_mean.png','mean')
pl.plot_genes_meanvar_barchart(dfm,batch_label,dl.outpath+'_var.png','var')
# pl.plot_gene_depthvar_barchart(dfm,batch_label,dl.outpath+'_dpvar.png','var')
pl.plot_stats(dfm,batch_label,dl.outpath+'_stats.png')

Q3 = dfm.quantile(0.75)
IQR = Q3

threshold = 1.5
upper_bound = Q3 + threshold * IQR

for column in dfm.columns:
    outliers_upper = dfm[column] > upper_bound[column]

    dfm.loc[outliers_upper, column] = upper_bound[column]

pl.plot_stats(dfm,batch_label,dl.outpath+'_stats_outli.png')

'''


asap = ASAPNMF(adata=dl,tree_max_depth=tree_max_depth,num_factors=num_factors,downsample_pbulk=downsample_pseudobulk,downsample_size=downsample_size,num_batch=bn)
asap.generate_pseudobulk()
asap.filter_pbulk(5) 

barcodes = asap.get_barcodes()
batch_label = np.array([x.split('@')[1] for x in barcodes])

# batch_label = np.array([x.split('-')[0] for x in barcodes])

asap.get_psuedobulk_batchratio(np.array(batch_label))
analysis.plot_pbulk_batchratio(asap.pbulk_batchratio,dl.outpath+'_pb_batch_ratio.png')

plt.plot(asap.pbulk_ysum.mean(0))
plt.savefig(dl.outpath+'_pb_sample_mean.png');plt.close() 

# f = hf.File(dl.inpath+'.h5','r')
# ct = ['ct'+str(x) for x in f['pbmc']['cell_type']]
# f.close()
# asap.get_psuedobulk_batchratio(np.array(ct))
# df = pd.DataFrame(asap.pbulk_batchratio)
# df_top = analysis.get_topic_top_genes(df,top_n=9)
# df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
# # df_top[df_top>20] = 20
# sns.clustermap(df_top.T,cmap='viridis',row_cluster=False)
# plt.savefig(dl.outpath+'_pb_heatmap.png');plt.close()


asap.run_nmf()


model = np.load(sample_out+'_dcnmf.npz',allow_pickle=True)

# 
df_beta = pd.DataFrame(model['nmf_beta'].T)
df_beta.columns = dl.genes
df_top = analysis.get_topic_top_genes(df_beta.iloc[:,:],top_n=3)
df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
df_top[df_top>20] = 100
sns.clustermap(df_top.T,cmap='viridis')
plt.savefig(dl.outpath+'_beta.png');plt.close()

