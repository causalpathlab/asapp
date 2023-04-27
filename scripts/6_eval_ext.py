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
pc25n25_data = sys.argv[4]
pc25n50_data = sys.argv[5]
pc25n500_data = sys.argv[6]
phi_delta = str(sys.argv[7])
rho = sys.argv[8]
size = sys.argv[9]
seed = sys.argv[10]

phi = phi_delta.split('_')[0]
delta = phi_delta.split('_')[1]

result_file = nmf_data_path+'_eval_ext.csv'

def eval_model(model_data,mode):
    df = pd.read_csv(model_data)

    if df.shape[0]>10:
        df['cell_type'] = [x.split('_')[1] for x in df['cell']]

        score = normalized_mutual_info_score(df['cell_type'].values,df['cluster'].values)
        result = ['nmi',mode,phi,delta,rho,size,seed,score]
        df_res = pd.DataFrame(result).T
        if os.path.isfile(result_file):
            df_res.to_csv(result_file,index=False, mode='a',header=False)
        else:
            df_res.columns = ['method','mode','phi','delta','rho','size','seed','score']
            df_res.to_csv(result_file,index=False)

        df_match = df[['cell','cell_type','cluster']]
        df_match = df_match.groupby(['cell_type','cluster']).count().reset_index()
        df_match = df_match.pivot('cell_type','cluster')
        df_match = df_match.fillna(0).T
        df_match['assign'] = [x for x in df_match.max(axis=1)]
        score = 1 - df_match.apply(lambda x: sum(x[x != x.iloc[-1]]), axis=1).sum()/df.shape[0]

        result = ['purity',mode,phi,delta,rho,size,seed,score]
        df_res = pd.DataFrame(result).T
        if os.path.isfile(result_file):
            df_res.to_csv(result_file,index=False, mode='a',header=False)
        else:
            df_res.columns = ['method', 'mode','phi','delta','rho','size','seed','score']
            df_res.to_csv(result_file,index=False)

        result = ['leiden',mode,phi,delta,rho,size,seed,score]
        df_res = pd.DataFrame(result).T
        if os.path.isfile(result_file):
            df_res.to_csv(result_file,index=False, mode='a',header=False)
        else:
            df_res.columns = ['method', 'mode','phi','delta','rho','size','seed','score']
            df_res.to_csv(result_file,index=False)


eval_model(scanpy_data,'scanpy')
eval_model(pc25n25_data,'pc25n25')
eval_model(pc25n50_data,'pc25n50')
eval_model(pc25n500_data,'pc25n500')

