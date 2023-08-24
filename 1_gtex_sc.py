import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import h5py as hf
from scipy.sparse import csr_matrix,csc_matrix


##################
## generate GTEX specific tissue data
##################


# for tissue in ['Breast',
#  'Esophagus mucosa',
#  'Esophagus muscularis',
#  'Heart',
#  'Lung',
#  'Prostate',
#  'Skeletal muscle',
#  'Skin']:
    
f = hf.File('node_data/gtex_sc/raw/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad','r')  
pd.Series(f['obs']['Tissue']).value_counts().sort_index() 
list(f['obs']['__categories']['Tissue']) 

mtx_indptr = f['X']['indptr']
mtx_indices = f['X']['indices']
mtx_data = f['X']['data']

num_rows = len(f['obs']['_index'])
num_cols = len(f['var']['_index'])



rows = csr_matrix((mtx_data,mtx_indices,mtx_indptr),shape=(num_rows,num_cols))
mtx= rows.todense()

## get tissue level
codes = list(f['obs']['Tissue'])
cat = [x.decode('utf-8') for x in f['obs']['__categories']['Tissue']]
catd ={}
for ind,itm in enumerate(cat):catd[ind]=itm
tissue = [catd[x] for x in codes]

## get breast tissue index
breast_index = [ x for x,y in enumerate(tissue) if y == 'Breast']


## filter tissue data
mtx = mtx[breast_index,:]
smat = csr_matrix(mtx)

genes = f['var']['gene_ids']

mainf = f
sample='data/gtex_scbreast'
f = hf.File(sample+'.h5ad','w')


grp = f.create_group('obs')
grp.create_dataset('_index',data=mainf['obs']['_index'][breast_index])

grp = f.create_group('X')
grp.create_dataset('indptr',data=smat.indptr)
grp.create_dataset('indices',data=smat.indices)
grp.create_dataset('data',data=smat.data,dtype=np.int32)

grp = f.create_group('var')
g1 = grp.create_group('feature_name')
g1.create_dataset('categories',data=genes)

grp.create_dataset('shape',data=mtx.shape)
f.close()


##################
## generate ASAP input
##################

from asap.data import dataloader as dm

scd = dm.DataMergerTS('data/gtex_sc/')
scd.get_datainfo()
scd.merge_genes()
scd.merge_data('gtex_sc')



##################
## run ASAP
##################

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
from sklearn.preprocessing import StandardScaler
import h5py as hf

import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

plt.figure(figsize=(10,6))
plt.tight_layout()

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
df_top = analysis.get_topic_top_genes(df_beta.iloc[:,:],top_n=10)
df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
th = 25
df_top[df_top>25] = 25
sns.clustermap(df_top.T,cmap='viridis')
plt.savefig(dl.outpath+'_beta3.png');plt.close()


scaler = StandardScaler()
df_corr = pd.DataFrame(scaler.fit_transform(model['predict_corr']))
# df_corr = pd.DataFrame(model['predict_corr'])
df_corr.index = model['predict_barcodes']


## assign ids
df_umap= pd.DataFrame()
df_umap['cell'] =[x.split('@')[0] for x in df_corr.index.values]

## assign batch
batch_label = ([1 for x in df_corr.index.values])
df_umap['batch'] = batch_label

## assign topic
kmeans = KMeans(n_clusters=num_factors, init='k-means++',random_state=0).fit(df_corr)
df_umap['asap_topic'] = kmeans.labels_

## assign celltype
f = hf.File('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/data/gtex_sc/raw/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad','r')
cell_ids = [x.decode('utf-8') for x in f['obs']['_index']]
cat = [x for x in f['obs']['Broad cell type']]
f.close()
        

df_ct = pd.DataFrame(cat,columns=['celltype'])
df_ct['cell']= cell_ids

df_umap = pd.merge(df_umap,df_ct,on='cell',how='left')
df_umap = df_umap.drop_duplicates(subset='cell',keep=False)


########### pre bc

df_umap[['umap_1','umap_2']] = analysis.get2dprojection(df_corr.to_numpy(),mindist=0.4)


analysis.plot_umaps(df_umap,dl.outpath+'_pre_batchcorrection.png')