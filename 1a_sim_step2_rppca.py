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

from collections import Counter

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

def calculate_purity(true_labels, cluster_labels):
	cluster_set = set(cluster_labels)
	total_correct = sum(max(Counter(true_labels[i] for i, cl in enumerate(cluster_labels) if cl == cluster).values()) 
						for cluster in cluster_set)
	return total_correct / len(true_labels)


def _randomprojection(mtx,obs,depth,cluster_resolution):
	
	from asappy.projection import rpstruct
	
	rp = rpstruct.projection_data(depth,mtx.shape[1])
	rp_data = rpstruct.get_random_projection_data(mtx.T,rp)
	
	ct = getct(obs)
	
	_,cluster = asappy.leiden_cluster(rp_data,resolution=cluster_resolution)
	s1  = calculate_purity(ct,cluster)
	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3

def _pca(mtx,obs,pc_n):

	pca = PCA(n_components=pc_n)
	pc_data = pca.fit_transform(mtx)

	ct = getct(obs)

	_,cluster = asappy.leiden_cluster(pc_data,resolution=cluster_resolution)
	s1  = calculate_purity(ct,cluster)
	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3


sample = sys.argv[1]
rho = sys.argv[2]
depth = sys.argv[3]
size = sys.argv[4]
seed = sys.argv[5]
topic = int(sys.argv[6])
cluster_resolution = float(sys.argv[7])
result_file = './results/'+sample+'_rppca_eval.csv'
print(sample)

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/'
data_size = 25000
number_batches = 1
asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)



current_dsize = asap_object.adata.uns['shape'][0]
df = asap_object.adata.construct_batch_df(current_dsize)
var = df.columns.values
obs = df.index.values
mtx = df.to_numpy()
outpath = asap_object.adata.uns['inpath']


print('running random projection...')
######## random projection 
rp1_s1,rp1_s2,rp1_s3 = _randomprojection(mtx,obs,50,cluster_resolution)
rp2_s1,rp2_s2,rp2_s3 = _randomprojection(mtx,obs,25,cluster_resolution)
rp3_s1,rp3_s2,rp3_s3 = _randomprojection(mtx,obs,15,cluster_resolution)

########
print('running pca...')
pc1_s1,pc1_s2,pc1_s3 = _pca(mtx,obs,50)
pc2_s1,pc2_s2,pc2_s3 = _pca(mtx,obs,25)
pc3_s1,pc3_s2,pc3_s3 = _pca(mtx,obs,15)


model_list = ['pc15','pc25','pc50','rp15','rp25','rp50']
res_list1 = [pc1_s1,pc2_s1,pc3_s1,rp1_s1,rp2_s1,rp3_s1]
res_list2 = [pc1_s2,pc2_s2,pc3_s2,rp1_s2,rp2_s2,rp3_s2]
res_list3 = [pc1_s3,pc2_s3,pc3_s3,rp1_s3,rp2_s3,rp3_s3]
res = []
res = construct_res(model_list,res_list1,'Purity',res)
res = construct_res(model_list,res_list2,'NMI',res)
res = construct_res(model_list,res_list3,'ARI',res)

cols = ['method','model','rho','depth','size','seed','topic','res','score']
df_res = pd.DataFrame(res)
df_res.columns = cols

save_eval(df_res)
