import asappy
import sys


sample = str(sys.argv[1])
data_size = int(sys.argv[2])
n_topics = int(sys.argv[3])
cluster_resolution = float(sys.argv[4])
seed = int(sys.argv[5])
wdir = sys.argv[6]

# sample ='pbmc_t_8_r_1.0'
# n_topics=8
# cluster_resolution = 1.0
# wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/examples/pbmc/'

number_batches = 1
# asappy.create_asap_data(sample)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)


# current_dsize = asap_object.adata.uns['shape'][0]
df = asap_object.adata.construct_batch_df(data_size)
var = df.columns.values
obs = df.index.values
mtx = df.to_numpy()
outpath = asap_object.adata.uns['inpath']

import asapc
from sklearn.preprocessing import StandardScaler 	
	
mtx = mtx.T
nmf_model = asapc.ASAPdcNMF(mtx,n_topics,int(seed))
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
adata = an.AnnData(shape=(len(obs),len(hgvs)))
adata.obs_names = [ x for x in obs]
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


##### cluster and celltype umap
asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
# print(asap_adata.obs.cluster.value_counts())
# asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
# asappy.plot_umap(asap_adata,col='cluster')
asap_adata.write(wdir+'results/'+sample+'.h5asap_fullad')