

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


# sample = 'sim_r_0.9_s_1_sd_1'
# rho=0.9;size=100;seed=1
result_file = './results/'+sample+'_eval.csv'

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

asap_full_score = 1.0
full_f = './results/'+sample+'.h5asapad_full'
if os.path.isfile(full_f):
    asap_full = an.read_h5ad(full_f)
    asap_full_score =  calc_score(asap_full.obs['celltype'].values,asap_full.obs['cluster'].values)

rp1 = pd.read_csv('./results/'+sample+'_rp5.csv.gz')
rp1_score = calc_score(getct(rp1.cell.values),rp1.cluster.values)

rp2 = pd.read_csv('./results/'+sample+'_rp10.csv.gz')
rp2_score = calc_score(getct(rp2.cell.values),rp2.cluster.values)

rp3 = pd.read_csv('./results/'+sample+'_rp50.csv.gz')
rp3_score = calc_score(getct(rp3.cell.values),rp3.cluster.values)

pc1 = pd.read_csv('./results/'+sample+'_pc5.csv.gz')
pc1_score = calc_score(getct(pc1.cell.values),pc1.cluster.values)

pc2 = pd.read_csv('./results/'+sample+'_pc10.csv.gz')
pc2_score = calc_score(getct(pc2.cell.values),pc2.cluster.values)

pc3 = pd.read_csv('./results/'+sample+'_pc50.csv.gz')
pc3_score = calc_score(getct(pc3.cell.values),pc3.cluster.values)

liger = pd.read_csv('./results/'+sample+'_liger.csv.gz')
liger_score = calc_score(getct(liger.cell.values),liger.cluster.values)

baseline = pd.read_csv('./results/'+sample+'_baseline.csv.gz')
baseline_score = calc_score(getct(baseline.cell.values),baseline.cluster.values)

res_list = [asap_score,asap_full_score,pc1_score,pc2_score,pc3_score,liger_score,baseline_score,rp1_score,rp2_score,rp3_score]
mode_list = ['asap','asapF','pc5','pc10','pc50','liger','base','rp5','rp10','rp50']
df_res = construct_res(mode_list,res_list)

save_eval(df_res)