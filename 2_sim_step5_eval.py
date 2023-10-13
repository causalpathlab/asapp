

######################################################
##### asap pipeline
######################################################

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import anndata as an
import pandas as pd
import numpy as np
import sys
import os


sample = sys.argv[1]
rho = sys.argv[2]
phi = sys.argv[3]
delta = sys.argv[4]
size = sys.argv[5]
seed = sys.argv[6]

# sample = 'sim_r_0.0_p_0.0_d_1.0_s_1000_sd_1'
# rho = 0.0
# phi = 0.0
# delta = 1.0
# size = 1000
# seed = 1

result_file = './results/'+sample+'_eval.csv'

def calc_score(ct,cl):
    nmi =  normalized_mutual_info_score(ct,cl)
    ari = adjusted_rand_score(ct,cl)
    return nmi,ari

def getct(ids):
    ct = [ x.replace('@'+sample,'') for x in ids]
    ct = [ '-'.join(x.split('_')[1:]) for x in ct]
    return ct 

def save_eval(df_res):
    if os.path.isfile(result_file):
        raise ValueError('file already exists')
    else:
        df_res.to_csv(result_file,index=False)

def construct_res(model_list,res_list,method,res):
    for model,r in zip(model_list,res_list):
        res.append([method,model,rho,size,seed,r])
    return res

def kmeans_cluster(df,k):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(df) 
        return kmeans.labels_

# n_factors = 15
# asap = an.read_h5ad('./results/'+sample+'.h5asapad')
# asap_cluster = kmeans_cluster(asap.obsm['corr'],n_factors)
# asap_s1,asap_s2 =  calc_score(asap.obs['celltype'].values,asap_cluster)

# asap_full_s1 = 0
# asap_full_s2 = 0
# full_f = './results/'+sample+'.h5asapad_full'
# if os.path.isfile(full_f):
#     asap_full = an.read_h5ad(full_f)
#     asap_cluster_full = kmeans_cluster(asap_full.obsm['corr'],n_factors)
#     asap_full_s1,asap_full_s2 =  calc_score(asap_full.obs['celltype'].values,asap_cluster_full)

asap = an.read_h5ad('./results/'+sample+'.h5asapad')
asap_s1,asap_s2 =  calc_score(asap.obs['celltype'].values,asap.obs['cluster'].values)

asap_full_s1 = 0
asap_full_s2 = 0
full_f = './results/'+sample+'.h5asapad_full'
if os.path.isfile(full_f):
    asap_full = an.read_h5ad(full_f)
    asap_full_s1,asap_full_s2 =  calc_score(asap_full.obs['celltype'].values,asap_full.obs['cluster'].values)

rp1 = pd.read_csv('./results/'+sample+'_rp5.csv.gz')
rp1_s1,rp1_s2= calc_score(getct(rp1.cell.values),rp1.cluster.values)

rp2 = pd.read_csv('./results/'+sample+'_rp10.csv.gz')
rp2_s1,rp2_s2 = calc_score(getct(rp2.cell.values),rp2.cluster.values)

rp3 = pd.read_csv('./results/'+sample+'_rp50.csv.gz')
rp3_s1,rp3_s2= calc_score(getct(rp3.cell.values),rp3.cluster.values)

pc1 = pd.read_csv('./results/'+sample+'_pc5.csv.gz')
pc1_s1,pc1_s2 = calc_score(getct(pc1.cell.values),pc1.cluster.values)

pc2 = pd.read_csv('./results/'+sample+'_pc10.csv.gz')
pc2_s1,pc2_s2 = calc_score(getct(pc2.cell.values),pc2.cluster.values)

pc3 = pd.read_csv('./results/'+sample+'_pc50.csv.gz')
pc3_s1,pc3_s2 = calc_score(getct(pc3.cell.values),pc3.cluster.values)

# liger = pd.read_csv('./results/'+sample+'_liger.csv.gz')
# liger_s1,liger_s2 = calc_score(getct(liger.cell.values),liger.cluster.values)
liger_s1,liger_s2 = 0,0

# baseline = pd.read_csv('./results/'+sample+'_baseline.csv.gz')
# baseline_s1,baseline_s2 = calc_score(getct(baseline.cell.values),baseline.cluster.values)
baseline_s1,baseline_s2 = 0,0

model_list = ['asap','asapF','pc5','pc10','pc50','liger','base','rp5','rp10','rp50']
res_list1 = [asap_s1,asap_full_s1,pc1_s1,pc2_s1,pc3_s1,liger_s1,baseline_s1,rp1_s1,rp2_s1,rp3_s1]
res_list2 = [asap_s2,asap_full_s2,pc1_s2,pc2_s2,pc3_s2,liger_s2,baseline_s2,rp1_s2,rp2_s2,rp3_s2]
res = []
res = construct_res(model_list,res_list1,'NMI',res)
res = construct_res(model_list,res_list2,'ARI',res)

cols = ['method','model','rho','size','seed','score']
df_res = pd.DataFrame(res)
df_res.columns = cols

save_eval(df_res)