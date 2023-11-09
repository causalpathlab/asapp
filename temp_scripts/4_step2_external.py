import sys
import numpy as np
import pandas as pd

import asappy
import anndata as an

import scanpy as sc
import anndata as an

from sklearn.cluster import KMeans



def _ligerpipeline(mtx,var,obs,outpath,K,cluster_resolution):

	from sklearn.model_selection import train_test_split
	import pyliger

	adata = an.AnnData(mtx)
	dfvars = pd.DataFrame(var)
	dfobs = pd.DataFrame(obs)
	adata.obs = dfobs
	adata.var = dfvars
	adata.var.rename(columns={0:'gene'},inplace=True) 
	adata.obs.rename(columns={0:'cell'},inplace=True) 
	adata.obs.index.name = 'cell'
	adata.var.index.name = 'gene'

	test_size = 0.5
	adata_train, adata_test = train_test_split(adata, test_size=test_size, random_state=42)

	adata_train.uns['sample_name'] = 'train'
	adata_test.uns['sample_name'] = 'test'
	adata_list = [adata_train,adata_test]


	ifnb_liger = pyliger.create_liger(adata_list)

	pyliger.normalize(ifnb_liger)
	pyliger.select_genes(ifnb_liger)
	pyliger.scale_not_center(ifnb_liger)
	pyliger.optimize_ALS(ifnb_liger, k = K)
	pyliger.quantile_norm(ifnb_liger)

	# H_norm = pd.DataFrame(np.vstack([adata.obsm['H_norm'] for adata in ifnb_liger.adata_list]))
	# _,cluster = asappy.leiden_cluster(H_norm,resolution=0.3)
	# cluster = kmeans_cluster(H_norm,n_topics)

	pyliger.leiden_cluster(ifnb_liger,resolution=cluster_resolution)

	obs = list(ifnb_liger.adata_list[0].obs.cell.values) + list(ifnb_liger.adata_list[1].obs.cell.values)
	cluster = list(ifnb_liger.adata_list[0].obs.cluster.values) + list(ifnb_liger.adata_list[1].obs.cluster.values)
	H_norm = pd.DataFrame()
	
	H_norm['cell'] = obs        
	H_norm['cluster'] = cluster
	H_norm = H_norm[['cell','cluster']]
	H_norm.to_csv(outpath+'_liger.csv.gz',index=False, compression='gzip')
	

def _scanpy(mtx,obs,var,cluster_resolution):
	adata = an.AnnData(mtx)
	dfvars = pd.DataFrame(var)
	dfobs = pd.DataFrame(obs)
	adata.obs = dfobs
	adata.var = dfvars

	sc.pp.filter_cells(adata, min_genes=0)
	sc.pp.filter_genes(adata, min_cells=0)
	sc.pp.normalize_total(adata)
	sc.pp.log1p(adata)
	sc.pp.highly_variable_genes(adata)
	adata = adata[:, adata.var.highly_variable]
	sc.tl.pca(adata)

	# df = pd.DataFrame(adata.obsm['X_pca'])
	# _,cluster = asappy.leiden_cluster(df)
	# cluster = kmeans_cluster(df,n_topics)
	
	sc.pp.neighbors(adata)
	sc.tl.leiden(adata,resolution=cluster_resolution)
	df = pd.DataFrame()        
	df['cell'] = adata.obs[0].values        
	df['cluster'] = adata.obs.leiden.values
	df = df[['cell','cluster']]
	df.to_csv(outpath+'_scanpy.csv.gz',index=False, compression='gzip')

def _baseline(mtx,obs,var,cluster_resolution):
	from sklearn.decomposition import NMF
	model = NMF(n_components=n_topics, init='random', random_state=0)
	W = model.fit_transform(mtx)
	cluster = kmeans_cluster(W,n_topics)
	df = pd.DataFrame()
	df['cell'] = obs
	df['cluster'] = cluster
	df = df[['cell','cluster']]
	df.to_csv(outpath+'_baseline.csv.gz',index=False, compression='gzip')

def kmeans_cluster(df,k):
		kmeans = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(df)
		return kmeans.labels_

	
# sample = str(sys.argv[1])
# n_topics = int(sys.argv[2])
# cluster_resolution = float(sys.argv[3])
# wdir = sys.argv[4]

sample ='brca2'
n_topics=11
cluster_resolution = 0.1
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/examples/brca2/'


print(sample)

data_size = 25000
number_batches = 4
# asappy.create_asap_data(sample)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)


current_dsize = asap_object.adata.uns['shape'][0]
df = asap_object.adata.construct_batch_df(current_dsize)
var = df.columns.values
obs = df.index.values
mtx = df.to_numpy()
outpath = asap_object.adata.uns['inpath']


######## full external nmf model 
print('running liger...')
_ligerpipeline(mtx,var,obs,outpath,n_topics,cluster_resolution)

# ######## baseline 
print('running scanpy...')
_scanpy(mtx,obs,var,cluster_resolution)


