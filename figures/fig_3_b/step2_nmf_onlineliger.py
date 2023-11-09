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
	asap_data_size = data_size[0]
	asap_data_size = 25000
 
	if data_size[0]>asap_data_size: 
		number_batches = np.ceil(np.data_size[0]/asap_data_size)
  
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
def _ligerpipeline(data_file_p1,data_file_p2,K,cluster_resolution):

	import pyliger
	from anndata import read_h5ad

	adata1 = read_h5ad(updated_data_file_p1, backed='r+')
	adata1.var.rename(columns={'var':'gene'},inplace=True) 
	adata1.obs.rename(columns={'obs':'cell'},inplace=True) 
	adata1.obs.index.name = 'cell'
	adata1.var.index.name = 'gene'

	adata2 = read_h5ad(updated_data_file_p2, backed='r+')
	adata2.var.rename(columns={'var':'gene'},inplace=True) 
	adata2.obs.rename(columns={'obs':'cell'},inplace=True) 
	adata2.obs.index.name = 'cell'
	adata2.var.index.name = 'gene'

	
	adata_list = [adata1,adata2]

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
	
sample = 'sim_r_0.99_d_10000_s_2350_s_3'
rho = 0.99
depth = 10000
size = 2350
seed = 3
topic = 13
cluster_resolution = 1.0


# sample = sys.argv[1]
# rho = sys.argv[2]
# depth = sys.argv[3]
# size = int(sys.argv[4])
# seed = int(sys.argv[5])
# topic = int(sys.argv[6])
# cluster_resolution = float(sys.argv[7])

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
'''
idea:
for 20k , take 10k two files and read backed in liger



'''

data_file_p1 = wdir+'data/'+sample.replace('2350','1550')+'.h5'
data_file_p2 = wdir+'data/'+sample.replace('2350','770')+'.h5'

import h5py as hf
from scipy import sparse


f = hf.File(data_file_p1,'r')
dshape = tuple(f['matrix']['shape'])
sparse_matrix = csr_matrix((f['matrix']['data'], f['matrix']['indices'], f['matrix']['indptr']), shape=dshape).toarray()
var = [x.decode('utf-8') for x in list(f['matrix']['features']['id']) ]
obs = [x.decode('utf-8') for x in list(f['matrix']['barcodes']) ]
f.close()
adata = an.AnnData(X=sparse_matrix)
adata.obs = pd.DataFrame(obs,columns=['obs'])
adata.var = pd.DataFrame(var,columns=['var'])
adata.backed = True
adata.uns['sample_name'] = os.path.basename(data_file_p1).replace('.h5','')

updated_data_file_p1 = data_file_p1.replace('/data/','/results/')
updated_data_file_p1 = updated_data_file_p1.replace('h5','hdf5')
adata.write_h5ad(updated_data_file_p1)

f = hf.File(data_file_p2,'r')
dshape = tuple(f['matrix']['shape'])
sparse_matrix = csr_matrix((f['matrix']['data'], f['matrix']['indices'], f['matrix']['indptr']), shape=dshape)
var = [x.decode('utf-8') for x in list(f['matrix']['features']['id']) ]
obs = [x.decode('utf-8') for x in list(f['matrix']['barcodes']) ]
f.close()
adata = an.AnnData(X=sparse_matrix)
adata.obs = pd.DataFrame(obs,columns=['obs'])
adata.var = pd.DataFrame(var,columns=['var'])
adata.backed = True
adata.uns['sample_name'] = os.path.basename(data_file_p2).replace('.h5','')
updated_data_file_p2 = data_file_p2.replace('/data/','/results/')
updated_data_file_p2 = updated_data_file_p2.replace('h5','hdf5')
adata.write_h5ad(updated_data_file_p2)



f = hf.File('./results/PBMC_control.h5ad','r+')
def h5toh5ad(input_file):
	import anndata as ad
	import h5py

	input_h5_file = input_file
	output_h5ad_file = input_file+'.h5ad'

	with h5py.File(input_h5_file, "r") as h5file:
		data = h5file["data"][:]
		gene_names = h5file["gene_names"][:]
		cell_names = h5file["cell_names"][:]

	# Create an AnnData object
	adata = ad.AnnData(X=data, var=gene_names, obs=cell_names)

	# Optionally, you can add additional metadata to the AnnData object if available

	# Write the AnnData object to the output H5ad file
	adata.write(output_h5ad_file)

print('running liger....')
start_time = time.time()

liger_s1,liger_s2,liger_s3 = _ligerpipeline(data_file_p1,data_file_p2,topic,cluster_resolution)


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


'''
switch to online version for liger ...
'''