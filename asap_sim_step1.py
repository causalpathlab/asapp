######################################################
##### asap step 1 - create data and random projection
######################################################


######################################################
#####  create asap data
######################################################
sample = 'sim'
import asappy
asappy.create_asap_data(sample)

######################################################
#####  random projection analysis
######################################################
import asappy
import umap
import pandas as pd
import numpy as np
from asappy.clustering.leiden import leiden_cluster
from umap.umap_ import find_ab_params, simplicial_set_embedding

sample = 'sim'
data_size = 12000 #11530
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

rp_data = asappy.generate_randomprojection(asap_object,tree_depth=10)

rp = rp_data['full']
# rp = rp_data[0]['1_0_25000']['rp_data']

rp_index = asap_object.adata.load_datainfo_batch(1,0,11530)



df=pd.DataFrame(rp)
df.index = rp_index
snn,cluster = leiden_cluster(df,resolution=1.0,k=25)

## umap clustering
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
dfumap['cell'] = rp_index
dfumap.columns = ['umap1','umap2','cell']
dfumap['celltype'] = [ x.split('_')[2].replace('@sim','') for x in dfumap['cell']]
dfumap['cluster'] = cluster
dfumap['cluster'] = pd.Categorical(dfumap['cluster'])
outpath = asap_object.adata.uns['inpath']+'_1_rp'
asappy.plot_umap_df(dfumap,'celltype',outpath)
asappy.plot_umap_df(dfumap,'cluster',outpath)


dfrp = df.copy()
dfrp.columns = ['rp'+str(x) for x in dfrp.columns]
dfrp['celltype'] = [ x.split('_')[2].replace('@sim','') for x in dfrp.index.values]
asappy.plot_randomproj(dfrp,'celltype',asap_object.adata.uns['inpath'])
