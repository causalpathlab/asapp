import asappy
import sys


sample = str(sys.argv[1])
n_topics = int(sys.argv[2])
print(sample)

data_size = 250000
number_batches = 1

asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)



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

