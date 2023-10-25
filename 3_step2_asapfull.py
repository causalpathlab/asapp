import asappy
import sys


sample = str(sys.argv[1])
n_topics = int(sys.argv[2])
cluster_resolution = float(sys.argv[3])
wdir = sys.argv[4]
print(sample)

data_size = 250000
number_batches = 1

asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)



import asapc
import numpy as np
from sklearn.preprocessing import StandardScaler 		
mtx=asap_object.adata.X.T
nmf_model = asapc.ASAPdcNMF(mtx,n_topics)
nmfres = nmf_model.nmf()

scaler = StandardScaler()
beta_log_scaled = scaler.fit_transform(nmfres.beta_log)

total_cells = asap_object.adata.uns['shape'][0]

asap_object.adata.varm = {}
asap_object.adata.obsm = {}
asap_object.adata.uns['pseudobulk'] ={}
asap_object.adata.uns['pseudobulk']['pb_beta'] = nmfres.beta
asap_object.adata.uns['pseudobulk']['pb_theta'] = nmfres.theta


pred_model = asapc.ASAPaltNMFPredict(mtx,beta_log_scaled)
pred = pred_model.predict()
asap_object.adata.obsm['corr'] = pred.corr
asap_object.adata.obsm['theta'] = pred.theta

import anndata as an
hgvs = asap_object.adata.var.genes
adata = an.AnnData(shape=(len(asap_object.adata.obs.barcodes),len(hgvs)))
adata.obs_names = [ x for x in asap_object.adata.obs.barcodes]
adata.var_names = [ x for x in hgvs]

for key,val in asap_object.adata.uns.items():
    adata.uns[key] = val

adata.varm['beta'] = asap_object.adata.uns['pseudobulk']['pb_beta'] 
adata.obsm['theta'] = asap_object.adata.obsm['theta']
adata.obsm['corr'] = asap_object.adata.obsm['corr']

adata.write_h5ad(asap_object.adata.uns['inpath']+'.h5asap_full')

import asappy
import anndata as an
from pyensembl import ensembl_grch38

asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap_full')

# gn = []
# for x in asap_adata.var.index.values:
#     try:
#         g = ensembl_grch38.gene_by_id(x.split('.')[0]).gene_name 
#         gn.append(g)
#     except:
#         gn.append(x)

# asap_adata.var.index = gn


##### beta heatmap
# asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=30)


##### cluster and celltype umap
asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
# print(asap_adata.obs.cluster.value_counts())
# asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
# asappy.plot_umap(asap_adata,col='cluster')
asap_adata.write(wdir+'results/'+sample+'.h5asap_fullad')