######################################################
##### asap pipeline
######################################################

sample = 'sim'
data_size = 15000
number_batches = 1

######################################################
##### single cell nmf
######################################################
import asappy


asap_object = asappy.create_asap(sample,data_size,number_batches=number_batches)
asappy.generate_pseudobulk(asap_object,tree_depth=10)
asappy.asap_nmf(asap_object,num_factors=10)
asappy.save_model(asap_object)



######################################################
##### scanpy pipeline
######################################################

from asappy.util._scanpy import run_basic_pipeline  
scanpy_data_size = 11530
df = asap_object.adata.construct_batch_df(scanpy_data_size)
run_basic_pipeline('./results/'+sample,df) 


######################################################
##### analysis
######################################################

import asappy
import anndata as an


asap_adata = an.read_h5ad('./results/'+sample+'.h5asap')
asappy.plot_gene_loading(asap_adata,max_thresh=100)
asappy.leiden_cluster(asap_adata,k=10,mode='corr',resolution=0.1)
asappy.run_umap(asap_adata,distance='cosine',min_dist=0.1)

asappy.plot_umap(asap_adata,col='cluster')

### sim data
asap_adata.obs['celltype'] = [ x.split('_')[2] for x in asap_adata.obs.index.values]
asappy.plot_umap(asap_adata,col='celltype')

asap_adata.write('./results/'+sample+'.h5asapad')


