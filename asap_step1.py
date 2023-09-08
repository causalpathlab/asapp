######################################################
##### asap step 1 - create data and random projection
######################################################


######################################################
#####  create asap data
######################################################
sample = 'pbmc'
import asappy
asappy.create_asap_data(sample)

######################################################
#####  random projection analysis
######################################################
import asappy
import pandas as pd
import numpy as np
from asappy.clustering.leiden import leiden_cluster
from umap.umap_ import find_ab_params, simplicial_set_embedding

data_size = 25000
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

rp_data = asappy.generate_randomprojection(asap_object,tree_depth=10)

rp = rp_data['full']

rpindxs = asap_object.adata.obs.barcodes.values

df=pd.DataFrame(rp)
df.index = rpindxs
snn,cluster = leiden_cluster(df,resolution=0.5,k=10)

## umap clustering
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
dfumap['cell'] = rpindxs
dfumap.columns = ['umap1','umap2','cell']

df = pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_results/pbmc_results/pbmc_scanpy_label.csv.gz')
df['cell'] = [x.split('@')[0] for x in df['cell']]
leiden_map = {x:y for x,y in zip(df['cell'],df['leiden'])}

selected = [ x for x in dfumap.cell.values if x in leiden_map.keys()]
dfumap = dfumap[dfumap['cell'].isin(selected)]
dfumap['celltype'] = [ leiden_map[x] for x in dfumap['cell']]


dfumap['cluster'] = cluster
dfumap['cluster'] = pd.Categorical(dfumap['cluster'])
dfumap['celltype'] = pd.Categorical(dfumap['celltype'])
outpath = asap_object.adata.uns['inpath']+'_1_rp'

plot_umap_df(dfumap,'cluster',outpath)
plot_umap_df(dfumap,'celltype',outpath)


dfrp = df.copy()
dfrp.columns = ['rp'+str(x) for x in dfrp.columns]
dfrp.index = [x.split('@')[0] for x in dfrp.index.values]
dfrp = dfrp.loc[selected]
dfrp['celltype'] = dfumap['celltype'].values
asappy.plot_randomproj(dfrp,'celltype',asap_object.adata.uns['inpath'])
