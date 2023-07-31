import sys
sys.path.append('../')
from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.util import analysis
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import h5py as hf

plt.figure(figsize=(10,6))
plt.tight_layout()

experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id


tree_max_depth = 10
num_factors = 10
batch_size = 25000
downsample_pseudobulk = True
downsample_size = 100

dl = DataSet(sample_in,sample_out)
sample_list = dl.get_dataset_names()
dl.initialize_data(sample_list,batch_size)

print(dl.inpath)
print(dl.outpath)


# 
model = np.load(sample_out+'_dcnmf.npz',allow_pickle=True)

# 
df_beta = pd.DataFrame(model['nmf_beta'].T)
df_beta.columns = dl.genes
df_top = analysis.get_topic_top_genes(df_beta.iloc[:,:],top_n=10)
df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
# df_top[df_top>20] = 20
sns.clustermap(df_top.T,cmap='viridis')
plt.savefig(dl.outpath+'_beta.png');plt.close()


scaler = StandardScaler()
df_corr = pd.DataFrame(scaler.fit_transform(model['predict_corr']))
# df_corr = pd.DataFrame(model['predict_corr'])
# df_corr.index = model['predict_barcodes']
df_corr.index = dl.barcodes



## assign ids
df_umap= pd.DataFrame()
df_umap['cell'] = df_corr.index.values

## assign batch
batch_label = ([x.split('-')[0] for x in df_corr.index.values])
df_umap['batch'] = batch_label

## assign topic
kmeans = KMeans(n_clusters=num_factors, init='k-means++',random_state=0).fit(df_corr)
df_umap['asap_topic'] = kmeans.labels_

## assign celltype
f = hf.File(dl.inpath+'.h5','r')
ct = ['ct'+str(x) for x in f['pbmc']['cell_type']]
f.close()
df_umap['celltype'] = ct

########### pre bc

df_umap[['umap_1','umap_2']] = analysis.get2dprojection(df_corr.to_numpy())
analysis.plot_umaps(df_umap,dl.outpath+'_pre_batchcorrection.png')


############### post bc

from asap.util import batch_correction as bc 

df_corr_sc = pd.DataFrame(bc.batch_correction_scanorama(df_corr.to_numpy(),batch_label,alpha=0.01,sigma=15))
df_corr_sc.index = dl.barcodes

np.corrcoef(df_corr.iloc[:,0],df_corr_sc.iloc[:,0])


df_umap_sc = df_umap[['cell','asap_topic','batch','celltype']]
df_umap_sc[['umap_1','umap_2']] = analysis.get2dprojection(df_corr_sc.to_numpy())
analysis.plot_umaps(df_umap_sc,dl.outpath+'_post_batchcorrection.png')

