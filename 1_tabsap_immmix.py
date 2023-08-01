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
# df_top[df_top>20] = 20
sns.clustermap(df_top.T,cmap='viridis')
plt.savefig(dl.outpath+'_beta.png');plt.close()


# scaler = StandardScaler()
# df_corr = pd.DataFrame(scaler.fit_transform(model['predict_corr']))
df_corr = pd.DataFrame(model['predict_corr'])
df_corr.index = model['predict_barcodes']



######### minibatch for visualization

import random 

N = 1000
minib_i = random.sample(range(0,df_corr.shape[0]),N)
df_corr = df_corr.iloc[minib_i,:]


## assign ids
df_umap= pd.DataFrame()
df_umap['cell'] =[x.split('@')[0] for x in df_corr.index.values]

## assign batch
batch_label = ([x.split('@')[1] for x in df_corr.index.values])
df_umap['batch'] = batch_label

## assign topic
kmeans = KMeans(n_clusters=num_factors, init='k-means++',random_state=0).fit(df_corr)
df_umap['asap_topic'] = kmeans.labels_

## assign celltype
inpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/data/tabula_sapiens/'
cell_type = []
for ds in dl.dataset_list:
    
    f = hf.File(inpath+ds+'.h5ad','r')
    cell_ids = [x.decode('utf-8') for x in f['obs']['_index']]
    codes = list(f['obs']['cell_type']['codes'])
    cat = [x.decode('utf-8') for x in f['obs']['cell_type']['categories']]
    f.close()
    
    catd ={}
    for ind,itm in enumerate(cat):catd[ind]=itm
    
    cell_type = cell_type + [ [x,catd[y]] for x,y in zip(cell_ids,codes)]

df_ct = pd.DataFrame(cell_type,columns=['cell','celltype'])


df_umap = pd.merge(df_umap,df_ct,on='cell',how='left')
df_umap = df_umap.drop_duplicates(subset='cell',keep=False)

select_ct = list(df_umap.celltype.value_counts()[:10].index)
select_ct = ['T', 'B', 'NK', 'Dendritic' 'macrophage', 'fibroblast' , 'epithelial' ]

ct_map = {}
for ct in select_ct:
    for v in df_umap.celltype.values:
        if ct in v:
            ct_map[v]=ct

df_umap['celltype'] = [ ct_map[x]+'_'+y if  x  in ct_map.keys() else 'others' for x,y in zip(df_umap['celltype'],df_umap['batch'])]


#### fix number of cells in data

keep_index = np.where(np.isin(np.array([x.split('@')[0] for x in df_corr.index.values]), df_umap['cell'].values))[0]
df_corr = df_corr.iloc[keep_index,:]

########### pre bc

df_umap[['umap_1','umap_2']] = analysis.get2dprojection(df_corr.to_numpy())

df_umap.to_csv(dl.outpath+'_prebc_umap.csv.gz',compression='gzip')

############### post bc

from asap.util import batch_correction as bc 

df_corr_sc = pd.DataFrame(bc.batch_correction_scanorama(df_corr.to_numpy(),np.array(batch_label),alpha=0.001,sigma=15))
df_corr_sc.index = df_corr.index

np.corrcoef(df_corr.iloc[:,0],df_corr_sc.iloc[:,0])


df_umap_sc = df_umap[['cell','asap_topic','batch','celltype']]
df_umap_sc[['umap_1','umap_2']] = analysis.get2dprojection(df_corr_sc.to_numpy())
df_umap_sc.to_csv(dl.outpath+'_postbc_umap.csv.gz',compression='gzip')


### plots
df_umap = pd.read_csv(dl.outpath+'_prebc_umap.csv.gz')
df_umap_sc = pd.read_csv(dl.outpath+'_postbc_umap.csv.gz')

df_umap = df_umap.drop(columns=['Unnamed: 0'])
df_umap_sc = df_umap_sc.drop(columns=['Unnamed: 0'])

analysis.plot_umaps(df_umap,dl.outpath+'_pre_batchcorrection.png')
analysis.plot_umaps(df_umap_sc,dl.outpath+'_post_batchcorrection.png')

