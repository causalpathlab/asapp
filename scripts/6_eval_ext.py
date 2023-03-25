import sys
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score


sim_data_path = sys.argv[1]
nmf_data_path = sys.argv[2]
scanpy_data = sys.argv[3]
pc2n10_data = sys.argv[4]
pc10n100_data = sys.argv[5]
pc50n1000_data = sys.argv[6]
alpha = sys.argv[7]
rho = sys.argv[8]
depth = sys.argv[9]
size = sys.argv[10]
seed = sys.argv[11]

result_file = nmf_data_path+'_eval_ext.csv'

def eval_model(model_data,mode):
    df = pd.read_csv(model_data)

    if df.shape[0]>10:
        df['cell_type'] = [x.split('_')[1] for x in df['cell']]

        score = normalized_mutual_info_score(df['cell_type'].values,df['cluster'].values)
        result = [mode,alpha,rho,depth,size,seed,score]
        df_res = pd.DataFrame(result).T
        if os.path.isfile(result_file):
            df_res.to_csv(result_file,index=False, mode='a',header=False)
        else:
            df_res.columns = ['mode','alpha','rho','depth','size','seed','score']
            df_res.to_csv(result_file,index=False)


eval_model(scanpy_data,'scanpy')
eval_model(pc2n10_data,'pc2n10')
eval_model(pc10n100_data,'pc10n100')
eval_model(pc50n1000_data,'pc50n1000')

