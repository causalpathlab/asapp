# ######################################################
# ##### bulk setup
# ######################################################

bk_sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'


######################################################
##### transfer learning
######################################################

import asappy
import anndata as an
import pandas as pd
import numpy as np

### get single cell beta, theta, and pbtheta + bulk theta
asap_adata = an.read_h5ad('./results/'+sc_sample+'.h5asapad')
sc_beta = pd.DataFrame(asap_adata.uns['pseudobulk']['pb_beta'].T)
sc_beta.columns = asap_adata.var.index.values


sc_theta = pd.DataFrame(asap_adata.obsm['theta'])
sc_theta.index= asap_adata.obs.index.values


sc_pbtheta = pd.DataFrame(asap_adata.uns['pseudobulk']['pb_theta'])
sc_pbtheta.index = ['pb'+str(x) for x in sc_pbtheta.index.values]
sc_pbtheta.columns = ['t'+str(x) for x in sc_pbtheta.columns]

bulk_theta = pd.read_csv(outpath+'bulk_theta_lmfit.csv.gz')
bulk_theta.index = bulk_theta['Unnamed: 0']
bulk_theta.drop(columns=['Unnamed: 0'],inplace=True)
bulk_theta.columns = ['t'+str(x) for x in bulk_theta.columns]

########### normalization ############
sc_pbtheta = sc_pbtheta.div(sc_pbtheta.sum(axis=1), axis=0)
bulk_theta = bulk_theta.div(bulk_theta.sum(axis=1), axis=0)

asappy.plot_dmv_distribution(sc_pbtheta,outpath+'_sc_pbtheta_')
asappy.plot_dmv_distribution(bulk_theta,outpath+'_bulktheta_')

import matplotlib.pylab as plt
import seaborn as sns
sns.clustermap(sc_pbtheta);plt.savefig(outpath+'scpbtheta.png');plt.close()
sns.clustermap(bulk_theta);plt.savefig(outpath+'bulk_theta.png');plt.close()


df = pd.concat([sc_pbtheta,bulk_theta],axis=0,ignore_index=False)
batch = ['pb' if 'pb' in x else 'bulk' for x in df.index.values]

from asappy.clustering import leiden_cluster

snn,cluster = leiden_cluster(df,resolution=0.1,k=10)

from umap.umap_ import find_ab_params, simplicial_set_embedding

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
dfumap['cell'] = df.index.values
dfumap.columns = ['umap1','umap2','cell']
dfumap['batch'] = ['pb' if 'pb' in x else 'bulk' for x in dfumap['cell']]
dfumap['cluster'] = cluster
dfumap['cluster'] = pd.Categorical(dfumap['cluster'])
dfumap['tissue'] = ['a' for x in sc_pbtheta.index.values] + [x.split('@')[1] for x in bulk_theta.index.values]
# outpath +='test'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/mix_'
asappy.plot_umap_df(dfumap,'batch',outpath)
asappy.plot_umap_df(dfumap,'cluster',outpath)
asappy.plot_umap_df(dfumap,'tissue',outpath)