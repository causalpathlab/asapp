######################################################
##### whole blood nmf
######################################################

sample = 'bloodbulk'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'+sample
######################################################
##### whole blood nmf
######################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py as hf

bf = 'gene_reads_2017-06-05_v8_breast_mammary_tissue.gct.gz'
df = pd.read_csv('node_data/gtex_bulk/'+bf,sep='\t',skiprows=2)
df = df[~df['Description'].str.contains('MT-')]
df = df[~df['Description'].str.contains('_RNA')]
df = df[~df['Description'].str.contains('_rRNA')]
df = df.drop(columns=['id','Description'])
df = df.T
df.columns = df.iloc[0,:]
df = df.iloc[1:,:]

df.columns = [x.split('.')[0] for x in df.columns]

# f = hf.File('node_results/pbmc_results/pbmc.h5','r')  
# sc_genes = [x.decode('utf-8') for x in f['pbmc']['genes']]
# f.close()

f = hf.File('data/gtex_scbreast.h5ad','r')  
sc_genes = [x.decode('utf-8') for x in f['var']['feature_name']['categories']]
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

np.savez(outpath+'_nmf',
        beta = nmf.beta,
        theta = reg.theta,
        corr = reg.corr)		


######################################################
##### bulk analysis
######################################################

import asappy
import anndata as an
import numpy as np
import pandas as pd

model = np.load(outpath+'_nmf.npz',allow_pickle=True)
asapadata = an.AnnData(shape=(model['theta'].shape[0],model['beta'].shape[0]))
asapadata.obs_names = [ 'b'+str(x) for x in range(model['theta'].shape[0])]
asapadata.var_names = pd.read_csv('./results/lungbulk_common_genes.csv').values.flatten()
asapadata.varm['beta'] = model['beta']
asapadata.obsm['theta'] = model['theta']
asapadata.obsm['corr'] = model['corr']
asapadata.uns['inpath'] = outpath

asappy.leiden_cluster(asapadata,k=10,mode='corr',resolution=0.4)
asappy.run_umap(asapadata,min_dist=0.4)
asappy.plot_umap(asapadata,col='cluster')
asappy.plot_gene_loading(asapadata)
asappy.plot_structure(asapadata,'theta')
asappy.plot_structure(asapadata,'corr')

ntopic =  pmf2topic(asapadata.varm['beta'],asapadata.obsm['theta'])
asapadata.obsm['theta_n'] = ntopic['prop']
asappy.plot_structure(asapadata,'theta_n')


######################################################
##### pbmc single cell nmf
######################################################
import asappy
import pandas as pd
select_genes = list(pd.read_csv('./results/'+sample+'common_genes.csv').values.flatten())
asap_object = asappy.create_asap('gtex_scbreast',data_size= 35000,select_genes=select_genes)
asappy.generate_pseudobulk(asap_object,tree_depth=10)
asappy.asap_nmf(asap_object,num_factors=10)
asappy.save_model(asap_object)



######################################################
##### pbmc analysis
######################################################
import anndata as an

asapadata = an.read_h5ad('./results/'+sample+'.h5asap')
asappy.leiden_cluster(asapadata,k=10,mode='corr',resolution=0.1)
asappy.run_umap(asapadata,min_dist=0.4)
asappy.plot_umap(asapadata,col='cluster')
asappy.plot_gene_loading(asapadata)

import pandas as pd
df = pd.read_csv('./results/sc_scanpy_leiden_label.csv.gz')
df['cell'] = [x.split('@')[0] for x in df['cell']]
leiden_map = {x:y for x,y in zip(df['cell'],df['leiden'])}
asapadata.obs['leiden'] = [ leiden_map[x] if x in leiden_map.keys() else 'others' for x in asapadata.obs.index.values]
asappy.plot_umap(asapadata,col='leiden')


######################################################
##### mix analysis
######################################################

import asappy
import anndata as an
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler


# model = np.load(outpath+'_nmf.npz',allow_pickle=True)
# bulkadata = an.AnnData(shape=(model['theta'].shape[0],model['beta'].shape[0]))
# bulkadata.obs_names = [ 'b'+str(x) for x in range(model['theta'].shape[0])]
# bulkadata.var_names = pd.read_csv('./results/common_genes.csv').values.flatten()
# bulkadata.varm['beta'] = model['beta']
# bulkadata.obsm['theta'] = model['theta']
# bulkadata.obsm['corr'] = model['corr']
# bulkadata.uns['inpath'] = outpath



scadata = an.read_h5ad('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/pbmc.h5asap')

### bulk raw data
dfbulk = pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_data/gtex_bulk/gene_reads_2017-06-05_v8_whole_blood.gct.gz',sep='\t',skiprows=2)
dfbulk = dfbulk[~dfbulk['Description'].str.contains('MT-')]
dfbulk = dfbulk[~dfbulk['Description'].str.contains('_RNA')]
dfbulk = dfbulk[~dfbulk['Description'].str.contains('_rRNA')]
dfbulk = dfbulk.drop(columns=['id','Description'])
dfbulk = dfbulk.T
dfbulk.columns = dfbulk.iloc[0,:]
dfbulk = dfbulk.iloc[1:,:]
dfbulk.columns = [x.split('.')[0] for x in dfbulk.columns]
dfbulk = dfbulk[pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'+sample+'common_genes.csv').values.flatten()]
dfbulk = dfbulk.loc[:, ~dfbulk.columns.duplicated()]
dfbulk = dfbulk.astype(float)
dfbulk  = dfbulk.div(dfbulk.sum(1),axis=0) * 1e4

dfpb = pd.DataFrame(scadata.uns['pseudobulk']['pb_data']).T
dfpb.columns = dfbulk.columns
dfpb = dfpb.astype(float)
dfpb.index = ['pb-'+str(x) for x in range(dfpb.shape[0])]
dfpb  = dfpb.div(dfpb.sum(1),axis=0) * 1e4


outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/mix'


df = pd.concat([dfbulk,dfpb])
batchlabel = [x.split('-')[0] for x in df.index.values]
plot_stats(df,batchlabel,outpath+'_stats.png')



import scanpy as sc
import pandas as pd
import matplotlib.pylab as plt

adata = sc.AnnData(df, 
        df.index.to_frame(), 
        df.columns.to_frame())

# sc.pp.filter_cells(adata, min_genes=25)
# sc.pp.filter_genes(adata, min_cells=3)
# adata.var['mt'] = adata.var_names.str.startswith('MT-')  
# sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# adata = adata[adata.obs.n_genes_by_counts < 2500, :]
# adata = adata[adata.obs.pct_counts_mt < 5, :]
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# adata = adata[:, adata.var.highly_variable]
# sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
# sc.pp.scale(adata, max_value=10)

sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata)
plt.savefig(outpath+'_scanpy_pca.png');plt.close()


sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata,resolution=0.3)
sc.pl.umap(adata, color=['leiden'])
plt.savefig(outpath+'_scanpy_umapLEIDEN.png');plt.close()

adata.obs['batch'] = [x.split('-')[0] for x in df.index.values] 
sc.pl.umap(adata, color=['batch'])
plt.savefig(outpath+'_scanpy_umapLEIDENb.png');plt.close()



### pyliger

import anndata as an
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler


# model = np.load(outpath+'_nmf.npz',allow_pickle=True)
# bulkadata = an.AnnData(shape=(model['theta'].shape[0],model['beta'].shape[0]))
# bulkadata.obs_names = [ 'b'+str(x) for x in range(model['theta'].shape[0])]
# bulkadata.var_names = pd.read_csv('./results/common_genes.csv').values.flatten()
# bulkadata.varm['beta'] = model['beta']
# bulkadata.obsm['theta'] = model['theta']
# bulkadata.obsm['corr'] = model['corr']
# bulkadata.uns['inpath'] = outpath



scadata = an.read_h5ad('./results/gtex_scbreast.h5asap')

### bulk raw data
dfbulk = pd.read_csv('node_data/gtex_bulk/'+bf,sep='\t',skiprows=2)
dfbulk = dfbulk[~dfbulk['Description'].str.contains('MT-')]
dfbulk = dfbulk[~dfbulk['Description'].str.contains('_RNA')]
dfbulk = dfbulk[~dfbulk['Description'].str.contains('_rRNA')]
dfbulk = dfbulk.drop(columns=['id','Description'])
dfbulk = dfbulk.T
dfbulk.columns = dfbulk.iloc[0,:]
dfbulk = dfbulk.iloc[1:,:]
dfbulk.columns = [x.split('.')[0] for x in dfbulk.columns]
dfbulk = dfbulk[pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/bloodbulkcommon_genes.csv').values.flatten()]
dfbulk = dfbulk.loc[:, ~dfbulk.columns.duplicated()]
dfbulk = dfbulk.astype(float)
dfbulk  = dfbulk.div(dfbulk.sum(1),axis=0) * 1e4

dfpb = pd.DataFrame(scadata.uns['pseudobulk']['pb_data']).T
dfpb.columns = dfbulk.columns
dfpb = dfpb.astype(float)
dfpb.index = ['pb-'+str(x) for x in range(dfpb.shape[0])]
dfpb  = dfpb.div(dfpb.sum(1),axis=0) * 1e4

pbadata = an.AnnData(dfpb, 
        dfpb.index.to_frame(), 
        dfpb.columns.to_frame())
pbadata.var.rename(columns={0:'gene'},inplace=True) 
pbadata.obs.rename(columns={0:'cell'},inplace=True) 
pbadata.obs.index.name = 'cell'
pbadata.var.index.name = 'gene'
pbadata.uns['sample_name'] = 'pb'

blkadata = an.AnnData(dfbulk, 
        dfbulk.index.to_frame(), 
        dfbulk.columns.to_frame())
blkadata.var.rename(columns={0:'gene'},inplace=True) 
blkadata.obs.rename(columns={0:'cell'},inplace=True) 
blkadata.obs.index.name = 'cell'
blkadata.var.index.name = 'gene'
blkadata.uns['sample_name'] = 'bulk'

adata_list = [pbadata, blkadata]
ifnb_liger = pyliger.create_liger(adata_list)

pyliger.normalize(ifnb_liger)
pyliger.select_genes(ifnb_liger)
pyliger.scale_not_center(ifnb_liger)
pyliger.optimize_ALS(ifnb_liger, k = 10)
pyliger.quantile_norm(ifnb_liger)
pyliger.leiden_cluster(ifnb_liger, resolution=0.25)
pyliger.run_umap(ifnb_liger, distance = 'cosine', n_neighbors = 30, min_dist = 0.3)
all_plots = pyliger.plot_by_dataset_and_cluster(ifnb_liger, axis_labels = ['UMAP 1', 'UMAP 2'], return_plots = True)

import matplotlib.pylab as plt

i = 1
for plot in enumerate(all_plots):
        plt.savefig(outpath+'liger'+str(i)+'_.png');plt.close();i+=1