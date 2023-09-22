# ######################################################
# ##### bulk setup
# ######################################################

sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'+sample

# ######################################################
##### mix analysis
######################################################
import anndata as an
import pandas as pd
import numpy as np

scadata = an.read_h5ad('./results/gtex_sc.h5asap')

model = np.load(outpath+'_nmf.npz',allow_pickle=True)
bulkadata = an.AnnData(shape=(model['theta'].shape[0],model['beta'].shape[0]))
bulkadata.obs_names = [ 'b'+str(x) for x in range(model['theta'].shape[0])]
bulkadata.var_names = pd.read_csv('./results/bulkcommon_genes.csv').values.flatten()
bulkadata.varm['beta'] = model['beta']
bulkadata.obsm['theta'] = model['theta']
bulkadata.obsm['corr'] = model['corr']
bulkadata.uns['inpath'] = outpath

n_genes = 5
sc_beta = pd.DataFrame(scadata.varm['beta'].T)
sc_beta.columns = scadata.var.index.values
bulk_beta = pd.DataFrame(bulkadata.varm['beta'].T)
bulk_beta.columns = scadata.var.index.values

from asappy.util import analysis
sc_top_genes = np.unique(analysis.get_topic_top_genes(sc_beta,n_genes)['Gene'].values)
bulk_top_genes = np.unique(analysis.get_topic_top_genes(bulk_beta,n_genes)['Gene'].values)

top_genes = np.unique(list(sc_top_genes)+list(bulk_top_genes))
top_genes = bulk_top_genes

sc_beta = sc_beta[top_genes]
bulk_beta = bulk_beta[top_genes]

sc_beta.index = ['pb_'+str(x) for x in sc_beta.index.values]
bulk_beta.index = ['bulk_'+str(x) for x in bulk_beta.index.values]


correlation_results = pd.DataFrame(index=sc_beta.index, columns=bulk_beta.index)
for i, row1 in sc_beta.iterrows():
    for j, row2 in bulk_beta.iterrows():
        correlation_results.loc[i, j] = row1.corr(row2)
correlation_results = correlation_results.apply(pd.to_numeric)
sns.clustermap(correlation_results,cmap='viridis')
plt.savefig(outpath+'_correlation_mix.png');plt.close()


beta = pd.concat([sc_beta,bulk_beta])

import seaborn as sns
import matplotlib.pyplot as plt

sns.clustermap(sc_beta.T,cmap='viridis',col_cluster=False)
plt.savefig(outpath+'_betamix.png');plt.close()
