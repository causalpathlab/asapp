from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from collections import Counter
import anndata as an
import asappy
import pandas as pd
import numpy as np
import sys
import os


sample = str(sys.argv[1])
data_size = int(sys.argv[2])
n_topics = int(sys.argv[3])
cluster_resolution = float(sys.argv[4])
seed = float(sys.argv[5])
wdir = sys.argv[6]
print(sample)

# sample ='pbmc_t_8_r_0.1'
# n_topics=8
# cluster_resolution = 0.1
# wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/examples/pbmc/'

result_file = wdir+'results/'+sample+'_eval.csv'


def getct(ids,sample):
    
    dfid = pd.DataFrame(ids,columns=['cell'])
    dfl = pd.read_csv(wdir+'results/'+sample.split('_')[0]+'_label.csv.gz')
    dfjoin = pd.merge(dfl,dfid,on='cell',how='right')
    ct = dfjoin['celltype'].values
        
    return ct

def calculate_purity(true_labels, cluster_labels):
    cluster_set = set(cluster_labels)
    total_correct = sum(max(Counter(true_labels[i] for i, cl in enumerate(cluster_labels) if cl == cluster).values()) 
                        for cluster in cluster_set)
    return total_correct / len(true_labels)

def calc_score(ct,cl,sample):
    ct = getct(ct,sample)    
    nmi =  normalized_mutual_info_score(ct,cl)
    ari =  adjusted_rand_score(ct,cl)
    purity = calculate_purity(ct,cl)
    return nmi,ari,purity

def save_eval(df_res):
    if os.path.isfile(result_file):
        raise ValueError('file already exists')
    else:
        df_res.to_csv(result_file,index=False)

def construct_res(model_list,res_list,method,res):
    for model,r in zip(model_list,res_list):
        res.append([method,model,data_size,n_topics,cluster_resolution,seed,r])
    return res

asap = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')
ids = [ x.replace('@'+ sample,'') for x in asap.obs.index.values]
asap_s1,asap_s2,asap_s3 =  calc_score(ids,asap.obs['cluster'].values,sample)

asap_full_s1 = 0
asap_full_s2 = 0
asap_full_s3 = 0
full_f = wdir+'results/'+sample+'.h5asap_fullad'
if os.path.isfile(full_f):
    asap_full = an.read_h5ad(full_f)
    ids = [ x.replace('@'+ sample,'') for x in asap_full.obs.index.values]
    asap_full_s1,asap_full_s2,asap_full_s3 =  calc_score(ids,asap_full.obs['cluster'].values,sample)


liger = pd.read_csv(wdir+'results/'+sample+'_liger.csv.gz')
ids = [ x.replace('@'+ sample,'') for x in liger.cell.values]
liger_s1,liger_s2,liger_s3 = calc_score(ids,liger.cluster.values,sample)

scanpy = pd.read_csv(wdir+'results/'+sample+'_scanpy.csv.gz')
ids = [ x.replace('@'+ sample,'') for x in scanpy.cell.values]
scanpy_s1,scanpy_s2,scanpy_s3 = calc_score(ids,scanpy.cluster.values,sample)

baseline = pd.read_csv(wdir+'results/'+sample+'_baseline.csv.gz')
ids = [ x.replace('@'+ sample,'') for x in baseline.cell.values]
baseline_s1,baseline_s2,baseline_s3 = calc_score(ids,baseline.cluster.values,sample)

model_list = ['asap','asapF','scanpy','liger','base']
res_list1 = [asap_s1,asap_full_s1,scanpy_s1,liger_s1,baseline_s1]
res_list2 = [asap_s2,asap_full_s2,scanpy_s2,liger_s2,baseline_s2]
res_list3 = [asap_s3,asap_full_s3,scanpy_s3,liger_s3,baseline_s3]
res = []
res = construct_res(model_list,res_list1,'NMI',res)
res = construct_res(model_list,res_list1,'ARI',res)
res = construct_res(model_list,res_list2,'Purity',res)

cols = ['method','model','size','n_topics','res','seed','score']
df_res = pd.DataFrame(res)
df_res.columns = cols

save_eval(df_res)