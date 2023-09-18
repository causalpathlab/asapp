

######################################################
##### asap pipeline
######################################################

from sklearn.metrics import normalized_mutual_info_score
import anndata as an
import pandas as pd
import numpy as np
import sys
import os


sample = sys.argv[1]
rho = sys.argv[2]
size = sys.argv[3]
seed = sys.argv[4]

result_file = './results/'+sample+'_eval.csv'

# sample = 'sim_r_0.9_s_100_sd_1'
# rho=0.9;size=100;seed=1

def calc_score(ct,cl):
    return normalized_mutual_info_score(ct,cl)

def getct(ids):
    ct = [ x.replace('@'+sample,'') for x in ids]
    ct = [ '-'.join(x.split('_')[2:]) for x in ct]
    return ct 

def save_eval(df_res):
    if os.path.isfile(result_file):
        raise ValueError('file already exists')
    else:
        df_res.to_csv(result_file,index=False)

def construct_res(mode_list,res_list):
    res = []
    for mode,r in zip(mode_list,res_list):
        res.append([mode,rho,size,seed,r])
    cols = ['mode','rho','size','seed','score']
    df_res = pd.DataFrame(res)
    df_res.columns = cols
    return df_res


asap = an.read_h5ad('./results/'+sample+'.h5asapad')
asap_score =  calc_score(asap.obs['celltype'].values,asap.obs['cluster'].values)

asap_full = an.read_h5ad('./results/'+sample+'.h5asapad_full')
asap_full_score =  calc_score(asap_full.obs['celltype'].values,asap_full.obs['cluster'].values)

pc2 = pd.read_csv('./results/'+sample+'_pc2n10.csv.gz')
pc2_score = calc_score(getct(pc2.cell.values),pc2.cluster.values)

pc10 = pd.read_csv('./results/'+sample+'_pc10n100.csv.gz')
pc10_score = calc_score(getct(pc10.cell.values),pc10.cluster.values)

pc50 = pd.read_csv('./results/'+sample+'_pc50n1000.csv.gz')
pc50_score = calc_score(getct(pc50.cell.values),pc50.cluster.values)

liger = pd.read_csv('./results/'+sample+'_liger.csv.gz')
liger_score = calc_score(getct(liger.cell.values),liger.cluster.values)

scpy = pd.read_csv('./results/'+sample+'_scanpy.csv.gz')
scpy_score = calc_score(getct(scpy.cell.values),scpy.cluster.values)

res_list = [asap_score,asap_full_score,pc2_score,pc10_score,pc50_score,liger_score,scpy_score]
mode_list = ['asap','asapF','pc2','pc10','pc50','liger','scpy']
df_res = construct_res(mode_list,res_list)

save_eval(df_res)