import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


sim_data_path = sys.argv[1]
nmf_data_path = sys.argv[2]
altnmf_data = sys.argv[3]
dcnmf_data = sys.argv[4]
alpha = sys.argv[5]
rho = sys.argv[6]
depth = sys.argv[7]
size = sys.argv[8]
beta = 1/float(alpha)

data_rows = list(pd.read_csv(sim_data_path +'.rows.csv.gz' )['rows']) 
result_file = nmf_data_path+'_eval.csv'


def eval_model(mode_file,result_file):
    df_theta = pd.DataFrame(mode_file['corr'])
    df_umap= pd.DataFrame()
    df_umap['cell'] = data_rows
    df_umap['topic_bulk'] = [x for x in df_theta.iloc[:,:].idxmax(axis=1)]
    df_umap['cell_type'] = [x.split('_')[1] for x in df_umap['cell']]
    score = normalized_mutual_info_score(df_umap['cell_type'].values,df_umap['topic_bulk'].values)
    result = [alpha,beta,rho,depth,size,score]
    if os.path.isfile(result_file):
        pd.DataFrame(result).to_csv(result_file,index=False, mode='a')
    else:
        pd.DataFrame(result).to_csv(result_file,index=False)


alt_model = np.load(altnmf_data)
eval_model(alt_model,result_file)

dc_model = np.load(dcnmf_data)
eval_model(dc_model,result_file)