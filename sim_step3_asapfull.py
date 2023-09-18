import asappy
import sys


sample = sys.argv[1]
# sample = 'sim_p_0.8_d_0.1_r_0.9_s_1_sd_1'
print(sample)

data_size = 15000
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)



import asapc
import numpy as np
from sklearn.preprocessing import StandardScaler 		
mtx=asap_object.adata.X.T
nmf_model = asapc.ASAPdcNMF(mtx,13)
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

#### nmf analysis
from pyensembl import ensembl_grch38

asap_adata = an.read_h5ad('./results/'+sample+'.h5asap_full')

gn = []
for x in asap_adata.var.index.values:
    try:
        g = ensembl_grch38.gene_by_id(x.split('.')[0]).gene_name 
        gn.append(g)
    except:
        gn.append(x)

asap_adata.var.index = gn
asappy.leiden_cluster(asap_adata,k=10,mode='corr',resolution=0.1)
asappy.run_umap(asap_adata,distance='cosine',min_dist=0.1)
ct = [ x.replace('@sim','') for x in asap_adata.obs.index.values]
ct = [ '-'.join(x.split('_')[2:]) for x in ct]
asap_adata.obs['celltype'] = ct
asap_adata.write_h5ad('./results/'+sample+'.h5asapad_full')


# asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=10)
# asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=100)
# asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=1000)
# asappy.plot_umap(asap_adata,col='cluster')
# asappy.plot_umap(asap_adata,col='celltype')