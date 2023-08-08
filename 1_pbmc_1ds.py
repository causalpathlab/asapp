import sys
sys.path.append('../')
from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.util import analysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import h5py as hf

import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id


model = np.load(sample_out+'_dcnmf.npz',allow_pickle=True)

tree_max_depth = model['params'][()]['tree_max_depth']
num_factors = model['params'][()]['num_factors']
batch_size = model['params'][()]['batch_size']
downsample_pseudobulk = model['params'][()]['downsample_pseudobulk']
downsample_size = model['params'][()]['downsample_size']

dl = DataSet(sample_in,sample_out)
sample_list = dl.get_dataset_names()
dl.initialize_data(sample_list,batch_size)

print(dl.inpath)
print(dl.outpath)

# 
df_beta = pd.DataFrame(model['nmf_beta'].T)
df_beta.columns = dl.genes
df_top = analysis.get_topic_top_genes(df_beta.iloc[:,:],top_n=3)
df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
df_top[df_top>20] = 100
sns.clustermap(df_top.T,cmap='viridis')
plt.savefig(dl.outpath+'_beta.png');plt.close()



df_corr = pd.DataFrame(model['predict_corr'])
df_corr.index = dl.barcodes



batch_label = ([x.split('-')[0] for x in df_corr.index.values])

batches = set(batch_label)


for i,b in enumerate(batches):
    indxs = [i for i,v in enumerate(batch_label) if v ==b]
    dfm = df_corr.iloc[indxs,:]
    scaler = StandardScaler()
    dfm = scaler.fit_transform(dfm)

    if i ==0:
        upd_indxs = indxs
        upd_df = dfm
    else:
        upd_indxs += indxs
        upd_df = np.vstack((upd_df,dfm))

batch_label = np.array(batch_label)[upd_indxs]
df_corr = pd.DataFrame(upd_df)
df_corr.index = np.array(dl.barcodes)[upd_indxs]


## assign ids
df_umap= pd.DataFrame()
df_umap['cell'] = df_corr.index.values

## assign batch
# batch_label = ([x.split('-')[0] for x in df_corr.index.values])
df_umap['batch'] = batch_label

## assign topic
kmeans = KMeans(n_clusters=num_factors, init='k-means++',random_state=0).fit(df_corr)
df_umap['asap_topic'] = kmeans.labels_

# assign celltype
f = hf.File(dl.inpath+'.h5','r')
ct = ['ct'+str(x) for x in f['pbmc']['cell_type']]
ctn = [x.decode('utf-8') for x in f['pbmc']['cell_type_name']]

celltype = [ ctn[int(x.replace('ct',''))] for x in ct]
f.close()
# df_umap['celltype'] = np.array(celltype)
df_umap['celltype'] = np.array(celltype)[upd_indxs]


########### pre bc

df_umap[['umap_1','umap_2']] = analysis.get2dprojection(df_corr.to_numpy())
analysis.plot_umaps(df_umap,dl.outpath+'_pre_batchcorrection.png')


############### post bc

from asap.util import batch_correction as bc 

df_corr_sc = pd.DataFrame(bc.batch_correction_scanorama(df_corr.to_numpy(),batch_label,alpha=0.001,sigma=15))
df_corr_sc.index = df_corr.index

np.corrcoef(df_corr.iloc[:,0],df_corr_sc.iloc[:,0])


df_umap_sc = df_umap[['cell','asap_topic','batch','celltype']]
df_umap_sc[['umap_1','umap_2']] = analysis.get2dprojection(df_corr_sc.to_numpy())
analysis.plot_umaps(df_umap_sc,dl.outpath+'_post_batchcorrection.png')



# import anndata as an
# import scanpy as sc

# scdata = an.AnnData(df_corr.to_numpy())
# scdata.obs['celltype'] = df_umap['celltype'].values
# sc.pp.neighbors(scdata, n_neighbors=10, n_pcs=40)
# sc.tl.umap(scdata)
# sc.pl.umap(scdata,color=['celltype'])
# plt.savefig('test.png');plt.close()
