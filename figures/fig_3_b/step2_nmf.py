import sys
import numpy as np
import pandas as pd

import asappy
import asapc
import anndata as an
import scanpy as sc


from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn.preprocessing import StandardScaler 		
from sklearn.cluster import KMeans
from collections import Counter

import time

from memory_profiler import profile

import h5py as hf
from scipy.sparse import csr_matrix

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

@profile
def _asap(sample,data_size,wdir,n_topics,cluster_resolution):
	
	asappy.create_asap_data(sample,working_dirpath=wdir)
	
	number_batches = 1
	asap_data_size = 25000
 
	if data_size[0]>asap_data_size: 
		number_batches = np.ceil(data_size[0]/asap_data_size)
  
	asap_object = asappy.create_asap_object(sample=sample,data_size=asap_data_size,number_batches=number_batches,working_dirpath=wdir)
 
	asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale')
	asappy.asap_nmf(asap_object,num_factors=n_topics,seed=seed)

	asap_adata = asappy.generate_model(asap_object,return_object=True)
 	
	asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)

	ct = getct(asap_adata.obs.index.values)

	cluster = asap_adata.obs.cluster.values

	s1  = calculate_purity(ct,cluster)

	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3

@profile
def _ligerpipeline(data_file,K,cluster_resolution):

	from sklearn.model_selection import train_test_split
	import pyliger
 
	f = hf.File(data_file,'r')
	data_size = tuple(f['matrix']['shape'])
	mtx = csr_matrix((f['matrix']['data'], f['matrix']['indices'], f['matrix']['indptr']), shape=data_size).toarray()
	var = [x.decode('utf-8') for x in list(f['matrix']['features']['id']) ]
	obs = [x.decode('utf-8') for x in list(f['matrix']['barcodes']) ]
	f.close()


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
 
	pyliger.select_genes(ifnb_liger)
	
	gene_use = 2000
	ifnb_liger.adata_list[0].uns['var_gene_idx'] = ifnb_liger.adata_list[0].var['norm_var'].sort_values(ascending=False).index.values[:gene_use]
	ifnb_liger.adata_list[1].uns['var_gene_idx'] = ifnb_liger.adata_list[1].var['norm_var'].sort_values(ascending=False).index.values[:gene_use]
	ifnb_liger.var_genes = np.union1d(ifnb_liger.adata_list[0].uns['var_gene_idx'],ifnb_liger.adata_list[1].uns['var_gene_idx'])
  
	pyliger.scale_not_center(ifnb_liger)
	pyliger.optimize_ALS(ifnb_liger, k = K,rand_seed=seed)
	pyliger.quantile_norm(ifnb_liger)
	pyliger.leiden_cluster(ifnb_liger,resolution=cluster_resolution)

	obs = list(ifnb_liger.adata_list[0].obs.cell.values) + list(ifnb_liger.adata_list[1].obs.cell.values)
	cluster = list(ifnb_liger.adata_list[0].obs.cluster.values) + list(ifnb_liger.adata_list[1].obs.cluster.values)
	
	ct = getct(obs)

	s1  = calculate_purity(ct,cluster)

	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3
	

# sample = 'sim_r_0.99_d_10000_s_155_s_3'
# rho = 0.99
# depth = 10000
# size = 155
# seed = 3
# topic = 13
# cluster_resolution = 1.0


sample = sys.argv[1]
rho = sys.argv[2]
depth = sys.argv[3]
size = int(sys.argv[4])
seed = int(sys.argv[5])
topic = int(sys.argv[6])
cluster_resolution = float(sys.argv[7])

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_3_b/'
data_file = wdir+'data/'+sample+'.h5'


f = hf.File(data_file,'r')
data_size = list(f['matrix']['shape'])
f.close()


result_file = './results/'+sample+'_nmf_eval.csv'
print(sample)


print('running asap...')
start_time = time.time()

asap_s1,asap_s2,asap_s3 = _asap(sample,data_size,wdir,topic,cluster_resolution)

end_time = time.time()
asap_time = end_time - start_time


####### other models  ##########


print('running liger....')
start_time = time.time()

liger_s1,liger_s2,liger_s3 = _ligerpipeline(data_file,topic,cluster_resolution)


end_time = time.time()
liger_time = end_time - start_time


model_list = ['asap','liger']
res_list1 = [asap_s1,liger_s1]
res_list2 = [asap_s2,liger_s2]
res_list3 = [asap_s3,liger_s3]
res_list4 = [asap_time,liger_time]

res = []
res = construct_res(model_list,res_list1,'Purity',res)
res = construct_res(model_list,res_list2,'NMI',res)
res = construct_res(model_list,res_list3,'ARI',res)
res = construct_res(model_list,res_list4,'Time',res)

cols = ['method','model','rho','depth','size','seed','topic','res','score']
df_res = pd.DataFrame(res)
df_res.columns = cols

save_eval(df_res)
