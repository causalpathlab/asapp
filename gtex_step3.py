# ######################################################
# ##### bulk setup
# ######################################################

sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'+sample

# ######################################################
##### mix analysis
######################################################

### pyliger

import anndata as an
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler


scadata = an.read_h5ad('./results/gtex_sc.h5asap')

### bulk raw data
dfbulk = pd.read_csv(outpath+'.csv.gz')
dfbulk.set_index('Unnamed: 0',inplace=True)
dfbulk = dfbulk.astype(float)
dfbulk  = dfbulk.div(dfbulk.sum(1),axis=0) * 1e4

dfpb = pd.DataFrame(scadata.uns['pseudobulk']['pb_data']).T
dfpb.columns = dfbulk.columns
dfpb = dfpb.astype(float)
dfpb.index = ['pb-'+str(x) for x in range(dfpb.shape[0])]
dfpb  = dfpb.div(dfpb.sum(1),axis=0) * 1e4

pbadata = an.AnnData(dfpb, 
        dfpb.index.to_frame(), 
        dfpb.columns.to_frame())
pbadata.var.rename(columns={0:'gene'},inplace=True) 
pbadata.obs.rename(columns={0:'cell'},inplace=True) 
pbadata.obs.index.name = 'cell'
pbadata.var.index.name = 'gene'
pbadata.uns['sample_name'] = 'pb'

blkadata = an.AnnData(dfbulk, 
        dfbulk.index.to_frame(), 
        dfbulk.columns.to_frame())
blkadata.var.rename(columns={0:'gene'},inplace=True) 
blkadata.obs.rename(columns={0:'cell'},inplace=True) 
blkadata.obs.index.name = 'cell'
blkadata.var.index.name = 'gene'
blkadata.uns['sample_name'] = 'bulk'


adata_list = [pbadata, blkadata]

cd /home/BCCRC.CA/ssubedi/projects/experiments/liger/src
import pyliger

ifnb_liger = pyliger.create_liger(adata_list)

pyliger.normalize(ifnb_liger)
pyliger.select_genes(ifnb_liger)
pyliger.scale_not_center(ifnb_liger)
pyliger.optimize_ALS(ifnb_liger, k = 10)
pyliger.quantile_norm(ifnb_liger)
# pyliger.leiden_cluster(ifnb_liger, resolution=0.01,k=10)

df1 = pd.DataFrame(ifnb_liger.adata_list[0].obsm['H_norm'])
df1.index = [str(x)+'pb' for x in df1.index.values]
df2 = pd.DataFrame(ifnb_liger.adata_list[1].obsm['H_norm'])
df2.index = [str(x)+'bulk' for x in df2.index.values]

df3 = pd.concat([df1,df2])

# df3.to_csv(outpath+'liger_h_norm.csv.gz',compression='gzip')

# pyliger.run_umap(ifnb_liger, distance = 'euclidean', n_neighbors = 25, min_dist = 0.01)

# all_plots = pyliger.plot_by_dataset_and_cluster(ifnb_liger, axis_labels = ['UMAP 1', 'UMAP 2'], return_plots = True)

# # List of file names to save the plots
# file_names = [outpath+'plot12.png', outpath+'plot22.png']

# # You can also use different formats like 'pdf', 'svg', etc.
# file_formats = ['png', 'png']
# for i, plot in enumerate(all_plots):
#     # Construct the full file path including the format
#     file_path = file_names[i]
    
#     # Save the plot to the specified file path and format
#     plot.save(filename=file_path, format=file_formats[i])





# ifnb_liger.adata_list[0].write(outpath+'pb_liger.h5')
# ifnb_liger.adata_list[1].write(outpath+'bulk_liger.h5')

# dfliger0 = pd.DataFrame(ifnb_liger.adata_list[0].obsm['umap_coords'])
# dfliger0['celltype'] = 'asap_pb'

# dfliger1 = pd.DataFrame(ifnb_liger.adata_list[1].obsm['umap_coords'])
# dfliger1['celltype'] = [x.split('@')[1] for x in ifnb_liger.adata_list[1].obs.index.values] 

# dfliger = pd.concat([dfliger0,dfliger1])
# dfliger.columns = ['umap1','umap2','cell-type']

# import asappy
# df=pd.read_csv(outpath+'liger_h_norm_umap.csv')
# df.columns = ['cell','umap1','umap2']
# df['batch'] = ['pb' if 'pb' in x else 'bulk' for x in df['cell']]


# df['cluster'] = list(ifnb_liger.adata_list[0].obs['cluster'].values) + list(ifnb_liger.adata_list[1].obs['cluster'].values)

# df['cluster'] = pd.Categorical(df['cluster'])

# asappy.plot_umap_df(df,'batch',outpath)
# asappy.plot_umap_df(df,'cluster',outpath)


"""
modifiled from welch-lab/pyliger
"""

import numpy as np
from annoy import AnnoyIndex


import leidenalg
import igraph as ig
from scipy.sparse import csr_matrix
import numpy as np

import numpy as np


def run_ann(theta,k, num_trees=None):

    num_observations = theta.shape[0]
    # decide number of trees
    if num_trees is None:
        if num_observations < 100000:
            num_trees = 10
        elif num_observations < 1000000:
            num_trees = 20
        elif num_observations < 5000000:
            num_trees = 50
        else:
            num_trees = 100

    # build knn graph
    t = AnnoyIndex(theta.shape[1], 'angular')
    for i in range(num_observations):
        t.add_item(i, theta[i])
    t.build(num_trees)

    # create knn indices matrices
    theta_knn = np.vstack([t.get_nns_by_vector(theta[i], k) for i in range(num_observations)])
    return theta_knn


def build_igraph(snn):
    sources, targets = snn.nonzero()
    weights = snn[sources, targets]

    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph()
    g.add_vertices(snn.shape[0])
    g.add_edges(list(zip(sources, targets)))
    g.es['weight'] = weights

    return g

def compute_snn(knn, prune):
    """helper function to compute the SNN graph
    
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6280782/
    
    """
    # int for indexing
    knn = knn.astype(np.int32)

    k = knn.shape[1]
    num_cells = knn.shape[0]

    rows = np.repeat(list(range(num_cells)), k)
    columns = knn.flatten()
    data = np.repeat(1, num_cells * k)
    snn = csr_matrix((data, (rows, columns)), shape=(num_cells, num_cells))

    snn = snn @ snn.transpose()

    rows, columns = snn.nonzero()
    data = snn.data / (k + (k - snn.data))
    data[data < prune] = 0

    return csr_matrix((data, (rows, columns)), shape=(num_cells, num_cells))

def leiden_cluster(df,
                   resolution=1.0,
                   k=10,
                   prune=1 / 15,
                   random_seed=1,
                   n_iterations=-1,
                   n_starts=10):

    knn = run_ann(df.to_numpy(),k)
    snn = compute_snn(knn, prune=prune)

    g = build_igraph(snn)

    np.random.seed(random_seed)
    max_quality = -1
    for i in range(n_starts):  
        seed = np.random.randint(0, 1000)
        kwargs = {'weights': g.es['weight'], 'resolution_parameter': resolution, 'seed': seed}  
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, n_iterations=n_iterations, **kwargs)

        if part.quality() > max_quality:
            cluster = part.membership
            max_quality = part.quality()

    return snn, cluster


snn,cluster = leiden_cluster(df3,resolution=0.1,k=25)

from umap.umap_ import find_ab_params, simplicial_set_embedding

min_dist = 0.8
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
        data = df3.to_numpy(),
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

df = pd.DataFrame(umap_coords[0])
df['cell'] = df3.index.values
df.columns = ['umap1','umap2','cell']
df['batch'] = ['pb' if 'pb' in x else 'bulk' for x in df['cell']]
df['cluster'] = cluster
df['cluster'] = pd.Categorical(df['cluster'])
df['tissue'] = ['a' for x in dfpb.index.values] + [x.split('@')[1] for x in dfbulk.index.values]
# outpath +='test'
asappy.plot_umap_df(df,'batch',outpath)
asappy.plot_umap_df(df,'cluster',outpath)
asappy.plot_umap_df(df,'tissue',outpath)
