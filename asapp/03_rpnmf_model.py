from util._io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import fastsca
import logging
import _rpstruct as rp
import _pmf
from sklearn import preprocessing
from sklearn.metrics.cluster import contingency_matrix,silhouette_score
np.random.seed(42)

experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sca = fastsca.FASTSCA()
sca.config = args
sca.initdata()
sca.loaddata()
print(sca.data.mtx.shape)
fn = sca.config.home + sca.config.experiment +sca.config.output + sca.config.sample_id+'/'+sca.config.sample_id


def rum_model_with_gt():
    min_leaf = 1000
    max_depth = 10
    n_iter = 5

    clust_mix = []
    clust_val = []

    for iter in range(n_iter):

        print('starting...'+str(iter))

        if iter ==0:
            rp_mat = []
            for i in range(max_depth):
                rp_mat.append(np.random.normal(size = (sca.data.mtx.shape[1],1)).flatten())

            rp_mat = np.asarray(rp_mat)
            print(rp_mat.shape)

        tree = rp.StepTree(sca.data.mtx,rp_mat)
        tree.build_tree(min_leaf,max_depth)
        bulkd = tree.make_bulk()
        print(len(bulkd))
        sum = 0
        for k in bulkd.keys(): sum += len(bulkd[k])
        print(sum)


        bulk = {}
        for key, value in bulkd.items(): 
            bulk[key] = np.asarray(sca.data.mtx[value].sum(0))[0]
        df = pd.DataFrame.from_dict(bulk,orient='index')


        m = _pmf.PoissonMF(n_components=max_depth)
        m.fit(df.to_numpy())
        B = m.gamma_b
        T = m.gamma_t

        bulk_index = {}
        for key, value in bulkd.items(): 
            bulk_index[key] = value
        df_index = pd.DataFrame.from_dict(bulk_index,orient='index')
        

        df_gt = pd.read_csv(fn+'_metadata.csv.gz')
        cell_topic_assignment = []
        for indx,row in df_index.iterrows():
            for i in row[pd.Series(row).notna()].values:
                cell_topic_assignment.append([ indx, df_gt.loc[i,'cell'],df_gt.loc[i,'label']])
        df_rp = pd.DataFrame(cell_topic_assignment,columns=['tree_leaf','cell','label'])
        df_match = pd.DataFrame(contingency_matrix(df_rp.tree_leaf,df_rp.label))
        df_match = df_match.div(df_match.sum(axis=1), axis=0)
        clust_mix.append(df_match.max(1).values)

        clust_val.append(df_match.max(axis=1).sum()/df_match.shape[0])

        scaler = preprocessing.StandardScaler().fit(B)
        rp_mat = scaler.transform(B)

        print('completed...'+str(iter))

    pd.DataFrame(clust_mix).to_csv(fn+"_cluster_trace.csv.gz",compression='gzip',index=False)
    pd.DataFrame(clust_val).to_csv(fn+"_cluster_val_trace.csv.gz",compression='gzip',index=False)

    df_index.to_csv(fn+"_"+str(n_iter)+"_rp_bulk_index.csv.gz",compression='gzip',index=False)
    pd.DataFrame(B).to_csv(fn+"_"+str(n_iter)+"_beta.csv.gz",compression='gzip',index=False)

def rum_model_no_gt():
    min_leaf = 250
    max_depth = 5
    n_iter = 10

    clust_val = []

    for iter in range(n_iter):

        print('starting...'+str(iter))

        if iter ==0:
            rp_mat = []
            for i in range(max_depth):
                rp_mat.append(np.random.normal(size = (sca.data.mtx.shape[1],1)).flatten())

            rp_mat = np.asarray(rp_mat)
            print(rp_mat.shape)

        tree = rp.StepTree(sca.data.mtx,rp_mat)
        tree.build_tree(min_leaf,max_depth)
        bulkd = tree.make_bulk()
        print(len(bulkd))
        sum = 0
        for k in bulkd.keys(): sum += len(bulkd[k])
        print(sum)


        bulk = {}
        for key, value in bulkd.items(): 
            bulk[key] = np.asarray(sca.data.mtx[value].sum(0))[0]
        df = pd.DataFrame.from_dict(bulk,orient='index')


        m = _pmf.PoissonMF(n_components=max_depth)
        m.fit(df.to_numpy())
        B = m.gamma_b
        T = m.gamma_t

        bulk_index = {}
        for key, value in bulkd.items(): 
            bulk_index[key] = value
        df_index = pd.DataFrame.from_dict(bulk_index,orient='index')
        
        cell_topic_assignment = {}
        for indx,row in df_index.iterrows():
            for i in row[pd.Series(row).notna()].values: cell_topic_assignment[ sca.data.rows[int(i)]]=indx

        tree_label = [ cell_topic_assignment[x] for x in sca.data.rows]
        clust_val.append(silhouette_score(sca.data.mtx,tree_label))

        scaler = preprocessing.StandardScaler().fit(B)
        rp_mat = scaler.transform(B)

        print('completed...'+str(iter))

    pd.DataFrame(clust_val).to_csv(fn+"_cluster_val_trace.csv.gz",compression='gzip',index=False)
    df_index.to_csv(fn+"_"+str(n_iter)+"_rp_bulk_index.csv.gz",compression='gzip',index=False)
    pd.DataFrame(B).to_csv(fn+"_"+str(n_iter)+"_beta.csv.gz",compression='gzip',index=False)

rum_model_no_gt()