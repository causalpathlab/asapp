######################################################
##### asap step 1 - create data and random projection
######################################################


######################################################
#####  create asap data
######################################################
sample = 'tabula_sapiens'
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

data_size = 23000
number_batches = 10
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

for ds in asap_object.adata.uns['dataset_list']:
    asap_object.adata.uns['dataset_batch_size'][ds] = 1000

rp_data,rp_indxes = asappy.generate_randomprojection(asap_object,tree_depth=10)


rp = rp_data[0]['1_0_23000']['rp_data']
for d in rp_data[1:]:
    rp = np.vstack((rp,d[list(d.keys())[0]]['rp_data']))

rpindxs = np.array(rp_indxes).flatten()




## assign celltype
import h5py as hf
inpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_data/tabula_sapiens/'
cell_type = []
for ds in asap_object.adata.uns['dataset_list']:
    
    f = hf.File(inpath+ds+'.h5ad','r')
    cell_ids = [x.decode('utf-8') for x in f['obs']['_index']]
    codes = list(f['obs']['cell_type']['codes'])
    cat = [x.decode('utf-8') for x in f['obs']['cell_type']['categories']]
    f.close()
    
    catd ={}
    for ind,itm in enumerate(cat):catd[ind]=itm
    
    cell_type = cell_type + [ [x,catd[y]] for x,y in zip(cell_ids,codes)]

df_ct = pd.DataFrame(cell_type,columns=['cell','celltype'])

rpindxs = [x.split('@')[0] for x in rpindxs]

df_cur = pd.read_csv(inpath+'tabula_sapiens_celltype_map.csv')
ctmap = {x:y for x,y in zip(df_cur['celltype'],df_cur['group'])}
df_ct['celltype2'] = [ctmap[x] if x in ctmap.keys() else 'others' for x in df_ct['celltype']]
df_ct = df_ct[df_ct['cell'].isin(rpindxs)]
ctmap = {x:y for x,y in zip(df_ct['cell'],df_ct['celltype2'])}


df=pd.DataFrame(rp)
df.index = rpindxs
snn,cluster = leiden_cluster(df,resolution=1.0,k=25)

## umap clustering
min_dist = 0.01
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
dfumap['cell'] = df.index.values
dfumap.columns = ['umap1','umap2','cell']
dfumap['celltype'] =  [ctmap[x] if x in ctmap.keys() else 'others' for x in dfumap['cell']]
dfumap['cluster'] = cluster
dfumap['cluster'] = pd.Categorical(dfumap['cluster'])
dfumap['celltype'] = pd.Categorical(dfumap['celltype'])
outpath = asap_object.adata.uns['inpath']+'_1_rp'
asappy.plot_umap_df(dfumap,'cluster',outpath)
asappy.plot_umap_df(dfumap,'celltype',outpath)


# dfrp = df.copy()
# dfrp.columns = ['rp'+str(x) for x in dfrp.columns]
# dfrp.index = [x.split('@')[0] for x in dfrp.index.values]
# dfrp = dfrp.loc[selected]
# dfrp['celltype'] = dfumap['celltype'].values
# asappy.plot_randomproj(dfrp,'celltype',asap_object.adata.uns['inpath'])
