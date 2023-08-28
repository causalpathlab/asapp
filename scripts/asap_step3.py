######################################################
##### asap pipeline
######################################################

sample = 'bcancerl'
data_size = 25000
number_batches = 13

######################################################
##### pbmc single cell nmf
######################################################
# import asappy


# asap_object = asappy.create_asap(sample,data_size,number_batches=number_batches)

# for ds in asap_object.adata.uns['dataset_list']:
#     asap_object.adata.uns['dataset_batch_size'][ds] = 2500

# asappy.generate_pseudobulk(asap_object,tree_depth=10)
# asappy.asap_nmf(asap_object,num_factors=10)
# asappy.save_model(asap_object)



######################################################
##### scanpy pipeline
######################################################

# from asappy.util._scanpy import run_basic_pipeline  
# scanpy_data_size = 25000
# df = asap_object.adata.construct_batch_df(scanpy_data_size)
# run_basic_pipeline('./results/'+sample,df) 


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

## tabula sapiens
# asap_adata.obs['tissuetype'] = [ x.split('@')[1] for x in asap_adata.obs.index.values]
# asappy.plot_umap(asap_adata,col='tissuetype')

asap_adata.write('./results/'+sample+'.h5asapad')
asap_adata = an.read_h5ad('./results/'+sample+'.h5asapad')


### breastcancer long 
metaf='/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_data/breastcancer/BRCA_GSE161529_CellMetainfo_table.tsv'
import pandas as pd
df = pd.read_csv(metaf,sep='\t')
df = df[['Cell','Celltype (major-lineage)']]
df.columns = ['cell','celltype']
df['cell'] = [x+'@bcancer' for x in df['cell']]
df.set_index('cell',inplace=True)
asap_adata.obs = pd.merge(asap_adata.obs,df,left_index=True,right_index=True,how='left')
asappy.plot_umap(asap_adata,col='celltype')



# ## gtex sc
# asap_adata.obs['celltype'] = [ x.split('-')[1].replace('@gtex_sc','') for x in asap_adata.obs.index.values]
# asappy.plot_umap(asap_adata,col='celltype')


# ##breast cancer
# import pandas as pd
# df = pd.read_csv('./results/1_d_celltopic_label.csv.gz')
# df['cell'] = [x.split('_')[0]+'_'+x.split('_')[1] for x in df['cell']]
# celltype_map = {x:y for x,y in zip(df['cell'],df['celltype'])}
# asap_adata.obs['celltype'] = [ celltype_map[x.split('@')[0]] if x.split('@')[0] in celltype_map.keys() else 'others' for x in asap_adata.obs.index.values]
# asappy.plot_umap(asap_adata,col='celltype')

# # ##pbmc
# import pandas as pd
# df = pd.read_csv('./results/pbmc_scanpy_label.csv.gz')
# df['cell'] = [x.split('@')[0] for x in df['cell']]
# leiden_map = {x:y for x,y in zip(df['cell'],df['leiden'])}
# asap_adata.obs['leiden'] = [ leiden_map[x] if x in leiden_map.keys() else 'others' for x in asap_adata.obs.index.values]
# asappy.plot_umap(asap_adata,col='leiden')


# asap_adata.obs['cluster'] = pd.Categorical(asap_adata.obs['cluster'])
# asap_adata.obs['leiden'] = ['l-'+str(x) for x in asap_adata.obs['leiden']]
# asap_adata.write('pbmc.h5asapad')

# import asappy
# import anndata as an
# import numpy as np
# asap_adata = an.read_h5ad('./results/pbmc.h5asapad')
# asappy.plot_umap(asap_adata,col='leiden')
# asappy.plot_umap(asap_adata,col='cluster')
# asappy.plot_gene_loading(asap_adata)
