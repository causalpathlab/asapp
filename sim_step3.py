

######################################################
##### asap pipeline
######################################################

sample = 'sim'
data_size = 15000
number_batches = 1


import asappy
import anndata as an
from sklearn.metrics import normalized_mutual_info_score

asap_adata = an.read_h5ad('./results/'+sample+'.h5asapad')
score = normalized_mutual_info_score(asap_adata.obs['celltype'].values,asap_adata.obs['cluster'].values)