# ######################################################
# ##### bulk setup
# ######################################################

sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'+sample


######################################################
##### single cell nmf
######################################################
data_size = 25000
number_batches = 9

import asappy
import pandas as pd
select_genes = list(pd.read_csv('./results/'+sample+'common_genes.csv').values.flatten())
asap_object = asappy.create_asap(sc_sample,data_size= data_size,select_genes=select_genes,number_batches=number_batches)
asappy.generate_pseudobulk(asap_object,tree_depth=10)
asappy.asap_nmf(asap_object,num_factors=10)
asappy.save_model(asap_object)



######################################################
##### sc analysis
######################################################

import asappy
import anndata as an


asap_adata = an.read_h5ad('./results/'+sc_sample+'.h5asap')
asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=50)
asappy.leiden_cluster(asap_adata,k=9,mode='corr',resolution=0.1)
asappy.run_umap(asap_adata,min_dist=0.1)
asap_adata.write('./results/'+sc_sample+'.h5asapad')

asap_adata = an.read_h5ad('./results/'+sc_sample+'.h5asapad')
asappy.plot_umap(asap_adata,col='cluster')

# ## gtex sc
asap_adata.obs['celltype'] = [ x.split('-')[1].replace('@gtex_sc','') for x in asap_adata.obs.index.values]
asappy.plot_umap(asap_adata,col='celltype')

## get compartment 

import h5py as hf
f = hf.File('./data/gtex_sc.h5ad')
codes = list(f['obs']['Cell types level 2'])
cat = [x.decode('utf-8') for x in f['obs']['__categories']['Cell types level 2']]
f.close()

catd ={}
for ind,itm in enumerate(cat):catd[ind]=itm
asap_adata.obs['celltype2'] = [ catd[x] for x in codes]
asappy.plot_umap(asap_adata,col='celltype2')
        