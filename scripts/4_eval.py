import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


sim_data_path = sys.argv[1]
nmf_data_path = sys.argv[2]
altnmf_data = sys.argv[3]
dcnmf_data = sys.argv[4]
fnmf_data = sys.argv[5]
alpha = sys.argv[6]
rho = sys.argv[7]
depth = sys.argv[8]
size = sys.argv[9]
seed = sys.argv[10]

data_rows = list(pd.read_csv(sim_data_path +'.rows.csv.gz' )['rows']) 
result_file = nmf_data_path+'_eval.csv'


def eval_model(model_file,result_file,mode):

    beta = model_file['beta']
    theta = model_file['theta']
    uu = beta.sum(0)
    beta = beta/uu
    prop = theta * uu
    zz = prop.T.sum(0).reshape(theta.shape[0],1)
    prop = prop/zz
    df_theta = pd.DataFrame(prop)
    df_theta.index = data_rows

    df_umap= pd.DataFrame()
    df_umap['cell'] = data_rows


    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=25, random_state=0).fit(df_theta.to_numpy())
    df_umap['topic_bulk'] = kmeans.labels_
    df_umap['cell_type'] = [x.split('_')[1] for x in df_umap['cell']]


    score = normalized_mutual_info_score(df_umap['cell_type'].values,df_umap['topic_bulk'].values)
    result = [mode,alpha,rho,depth,size,seed,score]
    df_res = pd.DataFrame(result).T
    if os.path.isfile(result_file):
        df_res.to_csv(result_file,index=False, mode='a',header=False)
    else:
        df_res.columns = ['mode','alpha','rho','depth','size','seed','score']
        df_res.to_csv(result_file,index=False)


alt_model = np.load(altnmf_data)
eval_model(alt_model,result_file,'alt')

dc_model = np.load(dcnmf_data)
eval_model(dc_model,result_file,'dc')

f_model = np.load(fnmf_data)
eval_model(f_model,result_file,'full')




