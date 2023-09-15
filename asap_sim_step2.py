######################################################
##### asap step 2 - pseudobulk analysis
######################################################

import asappy

sample = 'sim'
data_size = 12000
number_batches = 1

# asappy.create_asap_data(sample)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

normalization_raw='unitnorm'
normalization_pb='mscale'
z_high_gene_expression=10
z_high_gene_var=1.5

asappy.generate_pseudobulk(asap_object,tree_depth=10,normalization_raw=normalization_raw,normalization_pb=normalization_pb,z_high_gene_expression=z_high_gene_expression,z_high_gene_var=z_high_gene_var)

asappy.pbulk_cellcounthist(asap_object)

cell_index = asap_object.adata.load_datainfo_batch(1,0,11530)
ct = [ x.replace('@sim','') for x in cell_index]
ct = [ '-'.join(x.split('_')[2:]) for x in ct]
pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])


asappy.asap_nmf(asap_object,num_factors=13)

# from asappy.clustering.leiden import leiden_cluster
# from plotnine import *
# import pandas as pd 
# import numpy as np



# pmf2t = asappy.pmf2topic(beta=asap_object.adata.uns['pb_beta'] ,theta=asap_object.adata.uns['pseudobulk']['pb_theta'])
# df = pd.DataFrame(pmf2t['prop'])
# # df = pd.DataFrame(asap_object.adata.uns['pseudobulk']['pb_theta'])
# snn,cluster = leiden_cluster(df,resolution=1.0,k=13)


# df.columns = ['t'+str(x) for x in df.columns]
# df.reset_index(inplace=True)
# df['cluster'] = cluster

# dfm = pd.melt(df,id_vars=['index','cluster'])
# dfm.columns = ['id','cluster','topic','value']

# dfm['id'] = pd.Categorical(dfm['id'])
# dfm['cluster'] = pd.Categorical(dfm['cluster'])
# dfm['topic'] = pd.Categorical(dfm['topic'])

# col_vector = [
# '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
# '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
# '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
# ]

# p = (ggplot(data=dfm, mapping=aes(x='id', y='value', fill='topic')) +
#     geom_bar(position="stack", stat="identity", size=0) +
#     scale_fill_manual(values=col_vector) +
#     facet_grid('~ cluster', scales='free', space='free'))

# p = p + theme(
#     plot_background=element_rect(fill='white'),
#     panel_background = element_rect(fill='white'),
#     axis_text_x=element_blank())
# p.save(filename = asap_object.adata.uns['inpath']+'_pbulk_topic_struct.png', height=5, width=15, units ='in', dpi=300)

asappy.save_model(asap_object)


#### nmf analysis
import anndata as an
from pyensembl import ensembl_grch38

asap_adata = an.read_h5ad('./results/'+sample+'.h5asap')

gn = []
for x in asap_adata.var.index.values:
    try:
        g = ensembl_grch38.gene_by_id(x.split('.')[0]).gene_name 
        gn.append(g)
    except:
        gn.append(x)

asap_adata.var.index = gn

asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=10)


asappy.leiden_cluster(asap_adata,k=10,mode='corr',resolution=0.1)
asap_adata.obs.cluster.value_counts()
asappy.run_umap(asap_adata,distance='cosine',min_dist=0.1)

asappy.plot_umap(asap_adata,col='cluster')

ct = [ x.replace('@sim','') for x in asap_adata.obs.index.values]
ct = [ '-'.join(x.split('_')[2:]) for x in ct]
# remove_digits = str.maketrans('', '', '0123456789')
# ct = [s.translate(remove_digits) for s in ct]
asap_adata.obs['celltype'] = ct
asappy.plot_umap(asap_adata,col='celltype')

# pmf2t = asappy.pmf2topic(beta=asap_adata.varm['beta'] ,theta=asap_adata.obsm['theta'])
# asap_adata.obsm['theta_norm'] = pmf2t['prop']
# asappy.plot_structure(asap_adata,'theta_norm')


# sample = 'sim'
# data_size = 12000
# number_batches = 1
# asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)
# df = asap_object.adata.construct_batch_df(11530)
# df.columns = gn
# umap_coords= asap_adata.obsm['umap_coords']

# df_beta = pd.DataFrame(asap_adata.varm['beta'].T)
# df_beta.columns = asap_adata.var.index.values
# df_top = asappy.get_topic_top_genes(df_beta,2)
# marker_genes = df_top['Gene'].unique()
# plot_marker_genes(asap_adata.uns['inpath'],df,umap_coords,marker_genes,5,5)

# asap_adata.write('./results/'+sample+'.h5asapad')