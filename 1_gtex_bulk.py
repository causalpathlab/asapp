######################################################
##### whole blood nmf
######################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py as hf

outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/bulk_'

df = pd.read_csv('node_data/gtex_bulk/gene_tpm_2017-06-05_v8_whole_blood.gct.gz',sep='\t',skiprows=2)
df = df[~df['Description'].str.contains('MT-')]
df = df[~df['Description'].str.contains('_RNA')]
df = df[~df['Description'].str.contains('_rRNA')]
df = df.drop(columns=['id','Description'])
df = df.T
df.columns = df.iloc[0,:]
df = df.iloc[1:,:]

df.columns = [x.split('.')[0] for x in df.columns]

f = hf.File('node_results/pbmc_results/pbmc.h5','r')  
sc_genes = [x.decode('utf-8') for x in f['pbmc']['genes']]
f.close()
common_genes = [x for x in sc_genes if x in df.columns]
common_genes = list(set(common_genes))
common_genes.sort()
df = df[common_genes]
df = df.loc[:, ~df.columns.duplicated()]
pd.DataFrame(common_genes).to_csv(outpath+'common_genes.csv',index=False)


mtx = df.to_numpy().T
num_factors = 10

import asapc
nmf_model = asapc.ASAPdcNMF(mtx,num_factors)
nmf = nmf_model.nmf()

scaler = StandardScaler()
beta_log_scaled = scaler.fit_transform(nmf.beta_log)


reg_model = asapc.ASAPaltNMFPredict(mtx,beta_log_scaled)
reg = reg_model.predict()

np.savez(outpath+'bulk_nmf',
        beta = nmf.beta,
        theta = reg.theta,
        corr = reg.corr)		


######################################################
##### bulk analysis
######################################################

outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/bulk'
import asappy
import anndata as an

model = np.load(outpath+'_nmf.npz',allow_pickle=True)
asapadata = an.AnnData(shape=(model['theta'].shape[0],model['beta'].shape[0]))
asapadata.obs_names = [ 'b'+str(x) for x in range(model['theta'].shape[0])]
asapadata.var_names = pd.read_csv('./results/common_genes.csv').values.flatten()
asapadata.varm['beta'] = model['beta']
asapadata.obsm['theta'] = model['theta']
asapadata.obsm['corr'] = model['corr']
asapadata.uns['inpath'] = outpath

asappy.leiden_cluster(asapadata,k=10,mode='corr',resolution=0.8)
asappy.run_umap(asapadata,min_dist=0.4)
asappy.plot_umap(asapadata,col='cluster')
asappy.plot_gene_loading(asapadata)


######################################################
##### pbmc single cell nmf
######################################################
import asappy
import pandas as pd
select_genes = list(pd.read_csv('./results/common_genes.csv').values.flatten())
asap_object = asappy.create_asap('pbmc',data_size= 25000,select_genes=select_genes)
asappy.generate_pseudobulk(asap_object,tree_depth=10)
asappy.asap_nmf(asap_object,num_factors=10)
asappy.save_model(asap_object)



######################################################
##### pbmc analysis
######################################################
import anndata as an

asapadata = an.read_h5ad('./results/pbmc.h5asap')
asappy.leiden_cluster(asapadata,k=10,mode='corr',resolution=0.8)
asappy.run_umap(asapadata,min_dist=0.4)
asappy.plot_umap(asapadata,col='cluster')
asappy.plot_gene_loading(asapadata)

import pandas as pd
df = pd.read_csv('./results/pbmc_scanpy_label.csv.gz')
df['cell'] = [x.split('@')[0] for x in df['cell']]
leiden_map = {x:y for x,y in zip(df['cell'],df['leiden'])}
asapadata.obs['leiden'] = [ leiden_map[x] if x in leiden_map.keys() else 'others' for x in asapadata.obs.index.values]
asappy.plot_umap(asapadata,col='leiden')


######################################################
##### mix analysis
######################################################

outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/bulk'
import asappy
import anndata as an

model = np.load(outpath+'_nmf.npz',allow_pickle=True)
bulkadata = an.AnnData(shape=(model['theta'].shape[0],model['beta'].shape[0]))
bulkadata.obs_names = [ 'b'+str(x) for x in range(model['theta'].shape[0])]
bulkadata.var_names = pd.read_csv('./results/common_genes.csv').values.flatten()
bulkadata.varm['beta'] = model['beta']
bulkadata.obsm['theta'] = model['theta']
bulkadata.obsm['corr'] = model['corr']
bulkadata.uns['inpath'] = outpath


scadata = an.read_h5ad('./results/pbmc.h5asap')