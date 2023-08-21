
import asappy

asap_object = asappy.create_asap('pbmc',data_size= 10000)
asappy.generate_pseudobulk(asap_object,tree_depth=10)
asappy.asap_nmf(asap_object,num_factors=10)


###
import asappy
import anndata as an
import numpy as np
asap_adata = an.read_h5ad('./results/pbmc.h5asap')
asappy.louvain_cluster(asap_adata,k=10)