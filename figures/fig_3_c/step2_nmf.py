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
def _asap(sample,size,wdir,n_topics,cluster_resolution):
	
	asappy.create_asap_data(sample,working_dirpath=wdir)

	data_file = wdir+'results/'+sample+'.h5'
	f = hf.File(data_file,'r')
	rows,cols = 0,0
	for fk in f.keys():
		data_size = list(f[fk]['shape'])
		rows += data_size[0]
		cols = data_size[1]
	data_size = [rows,cols]
	f.close()

	number_batches = size_map[size]
  
	asap_object = asappy.create_asap_object(sample=sample,data_size=asap_data_size,number_batches=number_batches,working_dirpath=wdir)
 
	asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale')
 
	asappy.asap_nmf(asap_object,num_factors=n_topics,seed=seed)

	asap_adata = asappy.generate_model(asap_object,return_object=True)
 	
	# asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
	# ct = getct(asap_adata.obs.index.values)
	# cluster = asap_adata.obs.cluster.values
 
	mtx = asap_adata.obsm['corr']
	mtx = (mtx - np.mean(mtx, axis=0)) / np.std(mtx, axis=0, ddof=1)
	kmeans = KMeans(n_clusters=n_topics, init='k-means++',random_state=0).fit(mtx)
	cluster = kmeans.labels_
	ct = getct(asap_adata.obs.index.values)
 
	s1  = calculate_purity(ct,cluster)

	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3

# sample = 'sim_r_0.99_d_10000_s_15500_s_1'
# rho = 0.99
# depth = 10000
# size = 15500
# seed = 1
# topic = 13
# cluster_resolution = 1.0


sample = sys.argv[1]
rho = sys.argv[2]
depth = sys.argv[3]
size = int(sys.argv[4])
seed = int(sys.argv[5])
topic = int(sys.argv[6])
cluster_resolution = float(sys.argv[7])

wdir = 'experiments/asapp/figures/fig_3_c/'


# asap_data_size =  25000
# size_map = {
# 	15500 : 8,
# 	31000 : 16,
# 	46500 : 24,
# 	62000 : 32,
# 	77000 : 40
# }

# asap_data_size =  50000
# size_map = {
# 	15500 : 4,
# 	31000 : 8,
# 	46500 : 12,
# 	62000 : 16,
# 	77000 : 20
# }

# asap_data_size =  100000
# size_map = {
# 	15500 : 2,
# 	31000 : 4,
# 	46500 : 6,
# 	62000 : 8,
# 	77000 : 10
# }

asap_data_size =  200000
size_map = {
	15500 : 1,
	31000 : 2,
	46500 : 3,
	62000 : 4,
	77000 : 5
}


result_file = './results/'+sample+'_nmf_eval.csv'
print(sample)


print('running asap...')
start_time = time.time()

asap_s1,asap_s2,asap_s3 = _asap(sample,size,wdir,topic,cluster_resolution)

end_time = time.time()
asap_time = end_time - start_time


####### other models  ##########

model_list = ['asap']
res_list1 = [asap_s1]
res_list2 = [asap_s2]
res_list3 = [asap_s3]
res_list4 = [asap_time]

res = []
res = construct_res(model_list,res_list1,'Purity',res)
res = construct_res(model_list,res_list2,'NMI',res)
res = construct_res(model_list,res_list3,'ARI',res)
res = construct_res(model_list,res_list4,'Time',res)

cols = ['method','model','rho','depth','size','seed','topic','res','score']
df_res = pd.DataFrame(res)
df_res.columns = cols

save_eval(df_res)
