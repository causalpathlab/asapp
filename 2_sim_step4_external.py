import sys
import numpy as np
import pandas as pd

import asappy
import anndata as an

import scanpy as sc
import anndata as an

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score


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

def kmeans_cluster(df,k):
	kmeans = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(df) 
	return kmeans.labels_

def kmeans_cluster(df,k):
	kmeans = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(df)
	return kmeans.labels_

def fit_logreg(X,y):

    logistic_model = LogisticRegression(multi_class='ovr')

    predicted_probabilities = cross_val_predict(logistic_model, X, y, cv=5, method='predict_proba')

    classes = pd.Series(y).unique()
    
    binarized_labels = label_binarize(y, classes=classes)

    class_auc_values = []
    for ci in range(len(classes)):
            auc = roc_auc_score(binarized_labels[:, ci], predicted_probabilities[:, ci])
            class_auc_values.append(auc)

    return sum(class_auc_values) / len(class_auc_values)


# def _ligerpipeline(mtx,var,obs,outpath,K):

# 	from sklearn.model_selection import train_test_split
# 	import pyliger

# 	adata = an.AnnData(mtx)
# 	dfvars = pd.DataFrame(var)
# 	dfobs = pd.DataFrame(obs)
# 	adata.obs = dfobs
# 	adata.var = dfvars
# 	adata.var.rename(columns={0:'gene'},inplace=True) 
# 	adata.obs.rename(columns={0:'cell'},inplace=True) 
# 	adata.obs.index.name = 'cell'
# 	adata.var.index.name = 'gene'

# 	test_size = 0.5
# 	adata_train, adata_test = train_test_split(adata, test_size=test_size, random_state=42)

# 	adata_train.uns['sample_name'] = 'train'
# 	adata_test.uns['sample_name'] = 'test'
# 	adata_list = [adata_train,adata_test]


# 	ifnb_liger = pyliger.create_liger(adata_list)

# 	pyliger.normalize(ifnb_liger)
# 	pyliger.select_genes(ifnb_liger)
# 	pyliger.scale_not_center(ifnb_liger)
# 	pyliger.optimize_ALS(ifnb_liger, k = K)
# 	pyliger.quantile_norm(ifnb_liger)

# 	# H_norm = pd.DataFrame(np.vstack([adata.obsm['H_norm'] for adata in ifnb_liger.adata_list]))
# 	# _,cluster = asappy.leiden_cluster(H_norm)
# 	# cluster = kmeans_cluster(H_norm,n_topics)

# 	pyliger.leiden_cluster(ifnb_liger)

# 	obs = list(ifnb_liger.adata_list[0].obs.cell.values) + list(ifnb_liger.adata_list[1].obs.cell.values)
# 	cluster = list(ifnb_liger.adata_list[0].obs.cluster.values) + list(ifnb_liger.adata_list[1].obs.cluster.values)
# 	H_norm = pd.DataFrame()
	
# 	H_norm['cell'] = obs        
# 	H_norm['cluster'] = cluster
# 	H_norm = H_norm[['cell','cluster']]
# 	H_norm.to_csv(outpath+'_liger.csv.gz',index=False, compression='gzip')
    

# def _baseline(mtx,obs,var):
# 	adata = an.AnnData(mtx)
# 	dfvars = pd.DataFrame(var)
# 	dfobs = pd.DataFrame(obs)
# 	adata.obs = dfobs
# 	adata.var = dfvars

# 	sc.pp.filter_cells(adata, min_genes=0)
# 	sc.pp.filter_genes(adata, min_cells=0)
# 	sc.pp.normalize_total(adata)
# 	sc.pp.log1p(adata)
# 	sc.pp.highly_variable_genes(adata)
# 	adata = adata[:, adata.var.highly_variable]
# 	sc.tl.pca(adata)

# 	# df = pd.DataFrame(adata.obsm['X_pca'])
# 	# _,cluster = asappy.leiden_cluster(df)
# 	# cluster = kmeans_cluster(df,n_topics)
	
# 	sc.pp.neighbors(adata)
# 	sc.tl.leiden(adata)
# 	df = pd.DataFrame()        
# 	df['cell'] = adata.obs[0].values        
# 	df['cluster'] = adata.obs.leiden.values
# 	df = df[['cell','cluster']]
# 	df.to_csv(outpath+'_baseline.csv.gz',index=False, compression='gzip')


def _asap(asap_adata,cluster_resolution):
	corr = pd.DataFrame(asap_adata.obsm['corr'])
	ct = getct(asap_adata.obs.index.values)
	s1 = fit_logreg(corr,ct)

	asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
	s2,s3 =  calc_score(ct,asap_adata.obs['cluster'].values)
	return s1,s2,s3
	

def _randomprojection(mtx,obs,depth,cluster_resolution):
	
	from asappy.projection import rpstruct
	
	rp = rpstruct.projection_data(depth,mtx.shape[1])
	rp_data = rpstruct.get_random_projection_data(mtx.T,rp)
	
	ct = getct(obs)
	
	s1 = fit_logreg(rp_data,ct)

	_,cluster = asappy.leiden_cluster(pd.DataFrame(rp_data),resolution=cluster_resolution)
	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3

def _pca(mtx,obs,pc_n):

	pca = PCA(n_components=pc_n)
	pc_data = pca.fit_transform(mtx)

	ct = getct(obs)
	
	s1 = fit_logreg(pc_data,ct)

	_,cluster = asappy.leiden_cluster(pd.DataFrame(pc_data),resolution=cluster_resolution)
	s2,s3 = calc_score(ct,cluster)
	
	return s1,s2,s3


sample = sys.argv[1]
rho = sys.argv[2]
depth = sys.argv[3]
size = sys.argv[4]
seed = sys.argv[5]
topic = int(sys.argv[6])
cluster_resolution = float(sys.argv[7])
result_file = './results/'+sample+'_eval.csv'
print(sample)


asap_adata = an.read_h5ad('./results/'+sample+'.h5asap')
asap_s1, asap_s2, asap_s3 = _asap(asap_adata,cluster_resolution)

asapf_adata = an.read_h5ad('./results/'+sample+'.h5asap_full')
asapf_s1, asapf_s2, asapf_s3 = _asap(asapf_adata,cluster_resolution)

data_size = 25000
number_batches = 1
# asappy.create_asap_data(sample)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)



current_dsize = asap_object.adata.uns['shape'][0]
df = asap_object.adata.construct_batch_df(current_dsize)
var = df.columns.values
obs = df.index.values
mtx = df.to_numpy()
outpath = asap_object.adata.uns['inpath']


# print('running liger...')
######## full external nmf model 
# _ligerpipeline(mtx,var,obs,outpath,n_topics)
liger_s1, liger_s2, liger_s3 = 0,0,0

print('running random projection...')
######## random projection 
rp1_s1,rp1_s2,rp1_s3 = _randomprojection(mtx,obs,50,cluster_resolution)
rp2_s1,rp2_s2,rp2_s3 = _randomprojection(mtx,obs,10,cluster_resolution)
rp3_s1,rp3_s2,rp3_s3 = _randomprojection(mtx,obs,5,cluster_resolution)

########
print('running pca...')
pc1_s1,pc1_s2,pc1_s3 = _pca(mtx,obs,50)
pc2_s1,pc2_s2,pc2_s3 = _pca(mtx,obs,10)
pc3_s1,pc3_s2,pc3_s3 = _pca(mtx,obs,5)

# ######## baseline 
print('running baseline...')
# _baseline(mtx,obs,var)
baseline_s1, baseline_s2, baseline_s3 = 0,0,0

model_list = ['asap','asapF','pc5','pc10','pc50','liger','base','rp5','rp10','rp50']
res_list1 = [asap_s1,asapf_s1,pc1_s1,pc2_s1,pc3_s1,liger_s1,baseline_s1,rp1_s1,rp2_s1,rp3_s1]
res_list2 = [asap_s2,asapf_s2,pc1_s2,pc2_s2,pc3_s2,liger_s2,baseline_s2,rp1_s2,rp2_s2,rp3_s2]
res_list3 = [asap_s3,asapf_s3,pc1_s3,pc2_s3,pc3_s3,liger_s3,baseline_s3,rp1_s3,rp2_s3,rp3_s3]
res = []
res = construct_res(model_list,res_list1,'LR',res)
res = construct_res(model_list,res_list2,'NMI',res)
res = construct_res(model_list,res_list3,'ARI',res)

cols = ['method','model','rho','depth','size','seed','topic','res','score']
df_res = pd.DataFrame(res)
df_res.columns = cols

save_eval(df_res)



'''

'''