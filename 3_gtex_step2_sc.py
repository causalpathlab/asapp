# ######################################################
# ##### bulk setup
# ######################################################

sample = 'gtex_sc'


######################################################
##### single cell nmf
######################################################
data_size = 25000
number_batches = 9
K = 15

import asappy
import pandas as pd

select_genes = list(pd.read_csv('./results/bulkcommon_genes.csv').values.flatten())
asappy.create_asap_data(sample,select_genes=select_genes)

asappy.create_asap_data(sample)

asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)


normalize_pb='lscale'
hvg_selection=False
gene_mean_z=10
gene_var_z=2
normalize_raw=None

asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_raw=normalize_raw,normalize_pb=normalize_pb,hvg_selection=hvg_selection,gene_mean_z=gene_mean_z,gene_var_z=gene_var_z)

asappy.asap_nmf(asap_object,num_factors=K)
asappy.save_model(asap_object)



######################################################
##### sc analysis
######################################################

import asappy
import anndata as an


asap_adata = an.read_h5ad('./results/'+sample+'.h5asap')

asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=200)

asappy.leiden_cluster(asap_adata,k=10,mode='corr',resolution=0.1)
asappy.run_umap(asap_adata,min_dist=0.5)
asappy.plot_umap(asap_adata,col='cluster')

asap_adata.obs['celltype'] = [ x.split('-')[1].replace('@gtex_sc','') for x in asap_adata.obs.index.values]
asappy.plot_umap(asap_adata,col='celltype')

## get cell type level 2 
import h5py as hf
f = hf.File('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_data/gtex_sc/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad')
codes = list(f['obs']['Cell types level 2'])
cat = [x.decode('utf-8') for x in f['obs']['__categories']['Cell types level 2']]
f.close()

catd ={}
for ind,itm in enumerate(cat):catd[ind]=itm
asap_adata.obs['celltype2'] = [ catd[x] for x in codes]
asappy.plot_umap(asap_adata,col='celltype2')
        
asap_adata.write('./results/'+sample+'.h5asapad')