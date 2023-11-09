######################################################
##### single cell nmf
######################################################
import asappy
import sys

sample = str(sys.argv[1])
data_size = int(sys.argv[2])
n_topics = int(sys.argv[3])
cluster_resolution = float(sys.argv[4])
seed = int(sys.argv[5])
wdir = sys.argv[6]

print(sample)

asappy.create_asap_data(sample=sample,working_dirpath=wdir)

number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)
asappy.generate_pseudobulk(asap_object,tree_depth=10)
asappy.asap_nmf(asap_object,num_factors=n_topics,seed=seed)
asappy.generate_model(asap_object)

import anndata as an

asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')
asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
asap_adata.write(wdir+'results/'+sample+'.h5asapad')