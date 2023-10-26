# ######################################################
# ##### bulk setup
# ######################################################

sample = 'gtex_sc'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/examples/gtex/'


######################################################
##### single cell nmf
######################################################

import asappy
import pandas as pd

select_genes = list(pd.read_csv(wdir+'results/bulk_sc_common_genes.csv').values.flatten())
asappy.create_asap_data(sample,working_dirpath=wdir)

n_topics = 15
data_size = 30000
number_batches = 7

asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)
asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale',min_pseudobulk_size=500)
asappy.asap_nmf(asap_object,num_factors=n_topics)
asappy.save_model(asap_object)



######################################################
##### sc analysis
######################################################

import asappy
import anndata as an


asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')

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
        
asap_adata.write(wdir+'results/'+sample+'.h5asapad')