from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

######################################################
#####  create asap data
######################################################
# sample = 'pbmc_r_0.65_d_10000_s_250_s_2_t_8_r_0.1'
sample = 'sim_r_0.65_d_10000_s_250_s_2_t_13_r_0.1'
# sample = 'pbmc'
# sample = 'bc_80k'
import asappy

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/temp/'
data_size = 25000
number_batches = 1
asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
df = asap_object.adata.construct_batch_df(2000)
mtx = np.log1p(df.sample(n=100).to_numpy())
sns.clustermap(mtx)
plt.savefig(asap_object.adata.uns['inpath']+'_hmap.png');plt.close()

#####################################################
####  random projection analysis
#####################################################
import pandas as pd
import numpy as np
from asappy.clustering.leiden import leiden_cluster
from umap.umap_ import find_ab_params, simplicial_set_embedding


### get random projection
rp_data = asappy.generate_randomprojection(asap_object,tree_depth=10)
rp_data = rp_data['full']
rpi = asap_object.adata.load_datainfo_batch(1,0,rp_data.shape[0])
df=pd.DataFrame(rp_data)
df.index = rpi

# rp_data,rp_index = asappy.generate_randomprojection(asap_object,tree_depth=10)
# rp = rp_data[0]['1_0_25000']['rp_data']
# for d in rp_data[1:]:
#     rp = np.vstack((rp,d[list(d.keys())[0]]['rp_data']))
# rpi = np.array(rp_index[0])
# for i in rp_index[1:]:rpi = np.hstack((rpi,i))
# df=pd.DataFrame(rp)
# df.index = rpi

# from asappy.projection.rpstruct import projection_data
# rp_mat_list = projection_data(10,asap_object.adata.uns['shape'][1])
# mtx = asap_object.adata.load_data_batch(1,0,3250)
# mtx = mtx.T


### draw nearest neighbour graph and cluster
snn,cluster = leiden_cluster(df.to_numpy(),resolution=1.0)
pd.Series(cluster).value_counts() 



## umap cluster using neighbour graph
min_dist = 0.5
n_components = 2
spread: float = 1.0
alpha: float = 1.0
gamma: float = 1.0
negative_sample_rate: int = 5
maxiter = None
default_epochs = 500 if snn.shape[0] <= 10000 else 200
n_epochs = default_epochs if maxiter is None else maxiter
random_state = np.random.RandomState(42)

a, b = find_ab_params(spread, min_dist)

umap_coords = simplicial_set_embedding(
data = df.to_numpy(),
graph = snn,
n_components=n_components,
initial_alpha = alpha,
a = a,
b = b,
gamma = gamma,
negative_sample_rate = negative_sample_rate,
n_epochs = n_epochs,
init='spectral',
random_state = random_state,
metric = 'cosine',
metric_kwds = {},
densmap=False,
densmap_kwds={},
output_dens=False
)

dfumap = pd.DataFrame(umap_coords[0])
dfumap['cell'] = rpi
dfumap.columns = ['umap1','umap2','cell']

# ct = [ x.replace('@'+sample,'') for x in dfumap.cell.values]
# ct = [ '-'.join(x.split('_')[1:]) for x in ct]
# dfumap['celltype'] = pd.Categorical(ct)
# asappy.plot_umap_df(dfumap,'celltype',asap_object.adata.uns['inpath']+'_rp_')


f='/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_results/pbmc_results/pbmc_scanpy_label.csv.gz'
dfl = pd.read_csv(f)
lmap = {x.split('@')[0]:y  for x,y in zip(dfl['cell'].values,dfl['leiden'].values)}
dfumap['cell'] = [ x.split('@')[0] for x in dfumap['cell']]
ct = [lmap[x] if x in lmap.keys() else 'others' for x in dfumap.cell.values]
dfumap['celltype'] = pd.Categorical(ct)
asappy.plot_umap_df(dfumap,'celltype',asap_object.adata.uns['inpath']+'_rp_')


# f='/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_data/breastcancer/1_d_celltopic_label.csv.gz'
# dfl = pd.read_csv(f)
# dfl = dfl.loc[dfl['cell'].str.contains('GSE176078' ),:]
# dfl['cell'] = [ x.replace('_GSE176078','') for x in dfl['cell']]
# lmap = {x:y  for x,y in zip(dfl['cell'].values,dfl['celltype'].values)}
# dfumap['cell']= [ x.replace('@bc_80k','') for x in dfumap['cell']]
# dfumap['celltype'] = [lmap[x] if x in lmap.keys() else 'others' for x in dfumap.cell.values]
# dfumap['celltype']  = pd.Categorical(dfumap['celltype']  )
# asappy.plot_umap_df(dfumap,'celltype',asap_object.adata.uns['inpath']+'_rp_')


from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

def calc_score(ct,cl):
    nmi =  normalized_mutual_info_score(ct,cl)
    ari = adjusted_rand_score(ct,cl)
    return nmi,ari

asap_rp_s1,asap_rp_s2 =  calc_score([ str(x) for x in dfumap['celltype']],cluster)

print('NMI:'+str(asap_rp_s1))
print('ARI:'+str(asap_rp_s2))

#### plot rp1 to 10 scatter plot 

# dfrp = df.copy()
# dfrp.columns = ['rp'+str(x) for x in dfrp.columns]
# dfrp.index = [x.split('@')[0] for x in dfrp.index.values]
# dfrp['celltype'] = dfumap['celltype'].values
# asappy.plot_randomproj(dfrp,'celltype',asap_object.adata.uns['inpath'])

####################################################
###  pseudobulk  analysis
####################################################

asappy.generate_pseudobulk(asap_object,min_pseudobulk_size=100,tree_depth=10,normalize_pb='lscale')
asappy.pbulk_cellcounthist(asap_object)

# ct = [ x[0].replace('@'+sample,'') for x in asap_object.adata.obs.values]
# ct = [ '-'.join(x.split('_')[1:]) for x in ct]
pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])


# asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb=None,pseudobulk_filter=False)

# from asappy.util.analysis import get_topic_top_genes
# import matplotlib.pylab as plt
# import seaborn as sns
# import scanpy as sc
# import anndata as an
# import pandas as pd
# pb_data = asap_object.adata.uns['pseudobulk']['pb_data']

# adata = an.AnnData(pb_data)
# dfvars = pd.DataFrame([x for x in range(pb_data.shape[1])])
# dfobs = pd.DataFrame([x for x in range(pb_data.shape[0])])
# adata.obs = dfobs
# adata.var = dfvars

# ########### common

# sc.pp.filter_cells(adata, min_genes=25)
# sc.pp.filter_genes(adata, min_cells=3)
# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)

# dfn = adata.to_df()
# top_n = 100
# max_thresh = 10
# df_top = get_topic_top_genes(dfn,top_n)
# df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
# df_top[df_top>max_thresh] = max_thresh
# sns.clustermap(df_top.T,cmap='viridis')
# plt.savefig(asap_object.adata.uns['inpath']+'_pb_rawhmap.png');plt.close()

# ###########################################




from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

######################################################
##### asap step 2 - nmf and pseudobulk analysis
######################################################

import asappy
from asappy.clustering.leiden import leiden_cluster
from plotnine import *
import pandas as pd 
import numpy as np


# sample = 'sim_r_0.95_d_10000_s_250_s_2_t_13_r_0.1'
# sample = 'sim_r_0.65_d_10000_s_250_s_2_t_13_r_0.1'

sample = 'pbmc'
# sample = 'bc_80k'
data_size = 10000
number_batches = 1
K = 10
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/'

# asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)

asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale',min_pseudobulk_size=500)
# asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale',min_pseudobulk_size=500)

asappy.asap_nmf(asap_object,num_factors=K)
asappy.generate_model(asap_object)

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
import asappy
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
asappy.leiden_cluster(asap_adata,resolution=0.5)
print(asap_adata.obs.cluster.value_counts())
asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)

asappy.plot_umap(asap_adata,col='cluster')

f='/data/sishir/data/pbmc_211k/pbmc_label.csv.gz'
dfl = pd.read_csv(f)
lmap = {x:y  for x,y in zip(dfl['cell'].values,dfl['celltype'].values)}
asap_adata.obs.index = [ x.split('@')[0] for x in asap_adata.obs.index.values]
asap_adata.obs['celltype'] = [lmap[x] if x in lmap.keys() else 'others' for x in asap_adata.obs.index.values]
asap_adata.obs['celltype']  = pd.Categorical(asap_adata.obs['celltype']  )
asappy.plot_umap(asap_adata,col='celltype')


# ct = [ x.replace('@'+sample,'') for x in asap_adata.obs.index.values]
# ct = [ '-'.join(x.split('_')[1:]) for x in ct]
# asap_adata.obs['celltype'] = ct
# asappy.plot_umap(asap_adata,col='celltype')


# f='/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_data/breastcancer/1_d_celltopic_label.csv.gz'
# dfl = pd.read_csv(f)
# dfl = dfl.loc[dfl['cell'].str.contains('GSE176078' ),:]
# dfl['cell'] = [ x.replace('_GSE176078','') for x in dfl['cell']]
# lmap = {x:y  for x,y in zip(dfl['cell'].values,dfl['celltype'].values)}
# asap_adata.obs.index = [ x.replace('@bc_80k','') for x in asap_adata.obs.index.values]
# asap_adata.obs['celltype'] = [lmap[x] if x in lmap.keys() else 'others' for x in asap_adata.obs.index.values]
# asap_adata.obs['celltype']  = pd.Categorical(asap_adata.obs['celltype']  )
# asappy.plot_umap(asap_adata,col='celltype')


from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

def calc_score(ct,cl):
    nmi =  normalized_mutual_info_score(ct,cl)
    ari = adjusted_rand_score(ct,cl)
    return nmi,ari

asap_s1,asap_s2 =  calc_score([str(x) for x in asap_adata.obs['celltype'].values],asap_adata.obs['cluster'].values)

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
