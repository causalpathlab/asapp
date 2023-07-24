# %%
import sys
sys.path.append('../')
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
batch_size = 5000
downsample_pseudobulk = True
downsample_size = 100

dl = DataSet(sample_in,sample_out)
sample_list = dl.get_dataset_names()
dl.initialize_data(sample_list,batch_size)

print(dl.inpath)
print(dl.outpath)


# %%
model = np.load(sample_out+'_dcnmf.npz',allow_pickle=True)

# %%
df_beta = pd.DataFrame(model['nmf_beta'].T)
df_beta.columns = dl.genes

### if full data 
# df_corr = pd.DataFrame(model['corr'])
# df_corr.index = dl.barcodes


## if batch data 
scaler = StandardScaler()
df_corr = pd.DataFrame(scaler.fit_transform(model['predict_corr']))
df_corr.index = model['predict_barcodes']

# %%
df_top = analysis.get_topic_top_genes(df_beta.iloc[:,:],top_n=10)
df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
# df_top[df_top>20] = 20
sns.clustermap(df_top.T,cmap='viridis')
plt.savefig(dl.outpath+'beta.png');plt.close()
# %%
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df_umap= pd.DataFrame()
df_umap['cell'] = df_corr.index.values

# df_umap['topic_bulk'] = [x for x in df_corr.iloc[:,:].idxmax(axis=1)]

kmeans = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(df_corr)
df_umap['topic_bulk'] = kmeans.labels_

# umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.4,metric='cosine')
# proj_2d = umap_2d.fit(df_corr.iloc[:,:])
# df_umap[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]
# df_umap

# # df_umap = pd.read_csv(sample_out+'_theta_umap.csv')
# # df_umap.columns = ['cell','umap1','umap2']


# # %%
# cp = sns.color_palette(cc.glasbey_dark, n_colors=len(df_umap['topic_bulk'].unique()))
# p = sns.scatterplot(data=df_umap, x='umap1', y='umap2', hue='topic_bulk',s=5,palette=cp,legend=True)
# plt.legend(title='Topic',title_fontsize=18, fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
# # p.axes.set_title("topics from bulkNMF",fontsize=30)
# p.set_xlabel("UMAP1",fontsize=20)
# p.set_ylabel("UMAP2",fontsize=20)
# plt.savefig(dl.outpath+'theta_topic.png');plt.close()

# # %%
# df_umap['batch'] = [x.split('-')[0]for x in df_umap['cell']]
# cp = sns.color_palette(cc.glasbey_dark, n_colors=len(df_umap['batch'].unique()))
# p = sns.scatterplot(data=df_umap, x='umap1', y='umap2', hue='batch',s=5,palette=cp,legend=True)
# plt.legend(title='batch',title_fontsize=18, fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig(dl.outpath+'theta_batch.png');plt.close()






import bbknn.matrix
import scanpy as sc
batch_list = [x.split('-')[0]for x in df_corr.index]

import h5py as hf
f = hf.File('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/data/pbmc_1ds_2b/pbmc_1ds_2b.h5','r')
ct = ['ct'+str(x) for x in f['pbmc']['cell_type']]
f.close()

### treating correlation as data

# X = df_corr.to_numpy()
# adata = sc.AnnData(X,
# 	pd.DataFrame(model['predict_barcodes']),
# 	pd.DataFrame([i for i in range(X.shape[1])]))

# sc.tl.pca(adata, svd_solver='arpack')
# sc.pl.pca(adata)
# plt.savefig(dl.outpath+'_scanpy_pca.png');plt.close()


# adata.obs['batch_key'] = [x.split('-')[0]for x in model['predict_barcodes']]
# bbknn.bbknn(adata, batch_key='batch_key')


# sc.tl.umap(adata)
# sc.pl.umap(adata, color=['batch_key'])
# plt.savefig(dl.outpath+'theta_batch_bbknn.png');plt.close()

# adata.obs['topic'] = ['t'+str(t) for t in df_umap['topic_bulk']]

# sc.pl.umap(adata, color=['topic'])
# plt.savefig(dl.outpath+'theta_batch_bbknn_topic.png');plt.close()




# adata.obs['celltype'] = ct
# f.close()
# sc.pl.umap(adata, color=['celltype'])
# plt.savefig(dl.outpath+'theta_batch_bbknn_celltype.png');plt.close()

### treating correlation as pca 

X_pca = df_corr.to_numpy()
bbknn_out = bbknn.matrix.bbknn(X_pca, batch_list)	

X = np.zeros((X_pca.shape[0],len(dl.genes)))
adata = sc.AnnData(X,
	pd.DataFrame(model['predict_barcodes']),
	pd.DataFrame(dl.genes))

adata.obs['batch_key'] = [x.split('-')[0]for x in model['predict_barcodes']]

key_added = 'neighbors'
conns_key = 'connectivities'
dists_key = 'distances'
adata.uns[key_added] = {}
adata.uns[key_added]['params'] = bbknn_out[2]
adata.uns[key_added]['params']['use_rep'] = "X_pca"
adata.uns[key_added]['params']['bbknn']['batch_key'] = "batch_key"
adata.obsp[dists_key] = bbknn_out[0]
adata.obsp[conns_key] = bbknn_out[1]
adata.uns[key_added]['distances_key'] = dists_key
adata.uns[key_added]['connectivities_key'] = conns_key


adata.obsm['X_pca'] = X_pca

sc.tl.umap(adata)
sc.pl.umap(adata, color=['batch_key'])
plt.savefig(dl.outpath+'theta_batch_bbknn.png');plt.close()

adata.obs['topic'] = ['t'+str(t) for t in df_umap['topic_bulk']]

sc.pl.umap(adata, color=['topic'])
plt.savefig(dl.outpath+'theta_batch_bbknn_topic.png');plt.close()

adata.obs['celltype'] = ct
sc.pl.umap(adata, color=['celltype'])
plt.savefig(dl.outpath+'theta_batch_bbknn_celltype.png');plt.close()
