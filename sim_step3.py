

######################################################
##### asap pipeline
######################################################

sample = 'sim'
data_size = 15000
number_batches = 1


import asappy
import anndata as an

asap_adata = an.read_h5ad('./results/'+sample+'.h5asapad')