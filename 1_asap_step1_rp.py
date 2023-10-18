from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

######################################################
#####  create asap data
######################################################
sample = 'sim_r_0.95_d_1000_s_250_s_2_t_13_r_0.1'
# sample = 'pbmc'
import asappy


data_size = 25000
number_batches = 1
asappy.create_asap_data(sample)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

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
rp_index = asap_object.adata.load_datainfo_batch(1,0,rp_data.shape[0])
df=pd.DataFrame(rp_data)
df.index = rp_index

# from asappy.projection.rpstruct import projection_data
# rp_mat_list = projection_data(10,asap_object.adata.uns['shape'][1])
# mtx = asap_object.adata.load_data_batch(1,0,3250)
# mtx = mtx.T


### draw nearest neighbour graph and cluster
snn,cluster = leiden_cluster(df,resolution=0.5)
pd.Series(cluster).value_counts() 



## umap cluster using neighbour graph
min_dist = 0.1
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
dfumap['cell'] = rp_index
dfumap.columns = ['umap1','umap2','cell']

ct = [ x.replace('@'+sample,'') for x in dfumap.cell.values]
ct = [ '-'.join(x.split('_')[1:]) for x in ct]
dfumap['celltype'] = pd.Categorical(ct)
asappy.plot_umap_df(dfumap,'celltype',asap_object.adata.uns['inpath']+'_rp_')


# f='/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_results/pbmc_results/pbmc_scanpy_label.csv.gz'
# dfl = pd.read_csv(f)
# lmap = {x.split('@')[0]:y  for x,y in zip(dfl['cell'].values,dfl['leiden'].values)}
# dfumap['cell'] = [ x.split('@')[0] for x in dfumap['cell']]
# dfumap['celltype'] = [lmap[x] if x in lmap.keys() else 'others' for x in dfumap.cell.values]
# dfumap['celltype'] = pd.Categorical(dfumap['celltype'] )
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

# asappy.generate_pseudobulk(asap_object,min_pseudobulk_size=100,tree_depth=10,normalize_pb='lscale')
# asappy.pbulk_cellcounthist(asap_object)

# ct = [ x[0].replace('@'+sample,'') for x in asap_object.adata.obs.values]
# ct = [ '-'.join(x.split('_')[1:]) for x in ct]
# pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
# asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])


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
