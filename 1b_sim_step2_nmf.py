import sys
import numpy as np
import pandas as pd

import asappy
import anndata as an

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

import asapc
import scanpy as sc
import numpy as np
from sklearn.preprocessing import StandardScaler 		
from sklearn.cluster import KMeans
from collections import Counter


def calculate_purity(true_labels, cluster_labels):
    cluster_set = set(cluster_labels)
    total_correct = sum(max(Counter(true_labels[i] for i, cl in enumerate(cluster_labels) if cl == cluster).values()) 
                        for cluster in cluster_set)
    return total_correct / len(true_labels)

def calc_score(ct,cl):
    nmi =  normalized_mutual_info_score(ct,cl)
    ari = adjusted_rand_score(ct,cl)
    return nmi,ari

def getct(ids):
    ct = [ x.replace('@'+sample,'') for x in ids]
    ct = [ '-'.join(x.split('_')[1:]) for x in ct]
    return ct 

def save_eval(df_res):
	df_res.to_csv(result_file,index=False)

def construct_res(model_list,res_list,method,res):
    for model,r in zip(model_list,res_list):
        res.append([method,model,rho,depth,size,seed,topic,cluster_resolution,r])
    return res

def _asap(asap_object,n_topics,cluster_resolution):
	
	asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale',downsample_pseudobulk=False,pseudobulk_filter=False)
	asappy.asap_nmf(asap_object,num_factors=n_topics)

	asap_adata = asappy.save_model(asap_object,return_object=True)
 	
	asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)

	ct = getct(asap_adata.obs.index.values)

	cluster = asap_adata.obs.cluster.values

	s1  = calculate_purity(ct,cluster)

	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3


def _asapfull(asap_object,n_topics,cluster_resolution):

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

	hgvs = asap_object.adata.var.genes
	asap_adata = an.AnnData(shape=(len(asap_object.adata.obs.barcodes),len(hgvs)))
	asap_adata.obs_names = [ x for x in asap_object.adata.obs.barcodes]
	asap_adata.var_names = [ x for x in hgvs]

	for key,val in asap_object.adata.uns.items():
		asap_adata.uns[key] = val

	asap_adata.varm['beta'] = asap_object.adata.uns['pseudobulk']['pb_beta'] 
	asap_adata.obsm['theta'] = asap_object.adata.obsm['theta']
	asap_adata.obsm['corr'] = asap_object.adata.obsm['corr']

 	
	asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)

	ct = getct(asap_adata.obs.index.values)

	cluster = asap_adata.obs.cluster.values

	s1  = calculate_purity(ct,cluster)

	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3




def _ligerpipeline(mtx,var,obs,K,cluster_resolution):

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


	ifnb_liger = pyliger.create_liger(adata_list,remove_missing=False)
 
	pyliger.normalize(ifnb_liger)
	# pyliger.select_genes(ifnb_liger,var_thresh=1e-5)
	pyliger.select_genes(ifnb_liger)
	pyliger.scale_not_center(ifnb_liger)
	pyliger.optimize_ALS(ifnb_liger, k = K)
	pyliger.quantile_norm(ifnb_liger)
	pyliger.leiden_cluster(ifnb_liger,resolution=cluster_resolution)

	obs = list(ifnb_liger.adata_list[0].obs.cell.values) + list(ifnb_liger.adata_list[1].obs.cell.values)
	cluster = list(ifnb_liger.adata_list[0].obs.cluster.values) + list(ifnb_liger.adata_list[1].obs.cluster.values)
	
	ct = getct(obs)

	s1  = calculate_purity(ct,cluster)

	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3
	

def _scanpy(mtx,var,obs,cluster_resolution):
    
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

	
	sc.pp.neighbors(adata)
	sc.tl.leiden(adata,resolution=cluster_resolution)
 
	ct = getct(adata.obs[0].values)        
	cluster = adata.obs.leiden.values
 
	s1  = calculate_purity(ct,cluster)

	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3

def _baseline(mtx,var,obs,cluster_resolution):
	from sklearn.decomposition import NMF
	model = NMF(n_components=n_topics, init='random', random_state=0)
	W = model.fit_transform(mtx)
	cluster = kmeans_cluster(W,n_topics)

	ct = getct(obs)        

	s1  = calculate_purity(ct,cluster)

	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3

def kmeans_cluster(df,k):
		kmeans = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(df)
		return kmeans.labels_

	
sample = sys.argv[1]
rho = sys.argv[2]
depth = sys.argv[3]
size = sys.argv[4]
seed = sys.argv[5]
topic = int(sys.argv[6])
cluster_resolution = float(sys.argv[7])
result_file = './results/'+sample+'_nmf_eval.csv'
print(sample)

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/'
data_size = 25000
number_batches = 1
n_topics = 13
asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)


print('running asap...')
asap_s1,asap_s2,asap_s3 = _asap(asap_object,n_topics,cluster_resolution)

print('running asap full...')
asapfull_s1,asapfull_s2,asapfull_s3 = _asapfull(asap_object,n_topics,cluster_resolution)



current_dsize = asap_object.adata.uns['shape'][0]
df = asap_object.adata.construct_batch_df(current_dsize)
var = df.columns.values
obs = df.index.values
mtx = df.to_numpy()


print('running scanpy....')
scanpy_s1,scanpy_s2,scanpy_s3 = _scanpy(mtx,var,obs,cluster_resolution)

print('running baseline....')
baseline_s1,baseline_s2,baseline_s3 = _baseline(mtx,var,obs,cluster_resolution)

print('running liger....')
liger_s1,liger_s2,liger_s3 = _ligerpipeline(mtx,var,obs,n_topics,cluster_resolution)
print(liger_s1,liger_s2,liger_s3)


model_list = ['asap','asapf','liger','scanpy','baseline']
res_list1 = [asap_s1,asapfull_s1,liger_s1,scanpy_s1,baseline_s1]
res_list2 = [asap_s2,asapfull_s2,liger_s2,scanpy_s2,baseline_s2]
res_list3 = [asap_s3,asapfull_s3,liger_s3,scanpy_s3,baseline_s3]

res = []
res = construct_res(model_list,res_list1,'Purity',res)
res = construct_res(model_list,res_list2,'NMI',res)
res = construct_res(model_list,res_list3,'ARI',res)

cols = ['method','model','rho','depth','size','seed','topic','res','score']
df_res = pd.DataFrame(res)
df_res.columns = cols

save_eval(df_res)
