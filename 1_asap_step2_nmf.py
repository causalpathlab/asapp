######################################################
##### asap step 2 - nmf and pseudobulk analysis
######################################################

import asappy
from asappy.clustering.leiden import leiden_cluster
from plotnine import *
import pandas as pd 
import numpy as np

sample = 'sim_r_1.0_p_0.0_d_1.0_s_500_sd_1'
data_size = 25000
number_batches = 1
K = 13

asappy.create_asap_data(sample)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale',downsample_pseudobulk=False,pseudobulk_filter=False)

asappy.asap_nmf(asap_object,num_factors=K)
asappy.save_model(asap_object)

############ pseudobulk topic proportion

# pmf2t = asappy.pmf2topic(beta=asap_object.adata.uns['pseudobulk']['pb_beta'] ,theta=asap_object.adata.uns['pseudobulk']['pb_theta'])
# df = pd.DataFrame(pmf2t['prop'])
# snn,cluster = leiden_cluster(df,resolution=1.0,k=K)
# pd.Series(cluster).value_counts() 

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




################ nmf analysis

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


##### beta heatmap
asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=30)


##### cluster and celltype umap
asappy.leiden_cluster(asap_adata)
asap_adata.obs.cluster.value_counts()
asappy.run_umap(asap_adata,distance='cosine',min_dist=0.1)

asappy.plot_umap(asap_adata,col='cluster')

ct = [ x.replace('@'+sample,'') for x in asap_adata.obs.index.values]
ct = [ '-'.join(x.split('_')[1:]) for x in ct]
asap_adata.obs['celltype'] = ct
asappy.plot_umap(asap_adata,col='celltype')

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

def calc_score(ct,cl):
    nmi =  normalized_mutual_info_score(ct,cl)
    ari = adjusted_rand_score(ct,cl)
    return nmi,ari

asap_s1,asap_s2 =  calc_score(asap_adata.obs['celltype'].values,asap_adata.obs['cluster'].values)

print('NMI:'+str(asap_s1))
print('ARI:'+str(asap_s2))

# asap_adata.write('./results/'+sample+'.h5asapad')

# ####### marker genes
# df = asap_object.adata.construct_batch_df(11530)
# df = df.loc[:,asap_object.adata.uns['pseudobulk']['pb_hvgs']]
# df.columns = gn
# umap_coords= asap_adata.obsm['umap_coords']

# df_beta = pd.DataFrame(asap_adata.varm['beta'].T)
# df_beta.columns = asap_adata.var.index.values
# df_top = asappy.get_topic_top_genes(df_beta,2)
# marker_genes = df_top['Gene'].unique()[:25]
# plot_marker_genes(asap_adata.uns['inpath'],df,umap_coords,marker_genes,5,5)


# #### cells by factor plot 

# pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
# df = pd.DataFrame(pmf2t['prop'])

# df.columns = ['t'+str(x) for x in df.columns]
# df.reset_index(inplace=True)
# df['celltype'] = ct

# def sample_n_rows(group):
#     return group.sample(n=min(n, len(group)))

# n=25
# sampled_df = df.groupby('celltype', group_keys=False).apply(sample_n_rows)
# sampled_df.reset_index(drop=True, inplace=True)
# print(sampled_df)


# dfm = pd.melt(sampled_df,id_vars=['index','celltype'])
# dfm.columns = ['id','celltype','topic','value']

# dfm['id'] = pd.Categorical(dfm['id'])
# dfm['celltype'] = pd.Categorical(dfm['celltype'])
# dfm['topic'] = pd.Categorical(dfm['topic'])

# custom_palette = [
# "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
# "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
# "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
# "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
# '#6b6ecf', '#8ca252',  '#8c6d31', '#bd9e39', '#d6616b'
# ]

# p = (ggplot(data=dfm, mapping=aes(x='id', y='value', fill='topic')) +
#     geom_bar(position="stack", stat="identity", size=0) +
#     scale_fill_manual(values=custom_palette) +
#     facet_grid('~ celltype', scales='free', space='free'))

# p = p + theme(
#     plot_background=element_rect(fill='white'),
#     panel_background = element_rect(fill='white'),
#     axis_text_x=element_blank())
# p.save(filename = asap_adata.uns['inpath']+'_sc_topic_struct.png', height=5, width=15, units ='in', dpi=300)


# pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
# df = pd.DataFrame(asap_adata.obsm['corr'])

# df.columns = ['t'+str(x) for x in df.columns]
# df.reset_index(inplace=True)
# df['celltype'] = ct

# def sample_n_rows(group):
#     return group.sample(n=min(n, len(group)))

# n=25
# sampled_df = df.groupby('celltype', group_keys=False).apply(sample_n_rows)
# sampled_df.reset_index(drop=True, inplace=True)
# print(sampled_df)


# dfm = pd.melt(sampled_df,id_vars=['index','celltype'])
# dfm.columns = ['id','celltype','topic','value']

# dfm['id'] = pd.Categorical(dfm['id'])
# dfm['celltype'] = pd.Categorical(dfm['celltype'])
# dfm['topic'] = pd.Categorical(dfm['topic'])

# custom_palette = [
# "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
# "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
# "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
# "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
# '#6b6ecf', '#8ca252',  '#8c6d31', '#bd9e39', '#d6616b'
# ]

# p = (ggplot(data=dfm, mapping=aes(x='id', y='value', fill='topic')) +
#     geom_bar(position="stack", stat="identity", size=0) +
#     scale_fill_manual(values=custom_palette) +
#     facet_grid('~ celltype', scales='free', space='free'))

# p = p + theme(
#     plot_background=element_rect(fill='white'),
#     panel_background = element_rect(fill='white'),
#     axis_text_x=element_blank())
# p.save(filename = asap_adata.uns['inpath']+'_sc_topic_struct_corr.png', height=5, width=15, units ='in', dpi=300)
