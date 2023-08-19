import numpy as np
import pandas as pd
import logging
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
import random
logger = logging.getLogger(__name__)

def projection_data(depth,ndims):
    rp = []
    for _ in range(depth):
        rp.append(np.random.normal(size = (ndims,1)).flatten())                      
    return np.asarray(rp)
	

def get_projection_map(mtx,rp_mat):

    Z = np.dot(rp_mat,mtx)
    _, _, Q = randomized_svd(Z, n_components= Z.shape[0], random_state=0)
    
    scaler = StandardScaler()
    Q = scaler.fit_transform(Q.T)

    Q = (np.sign(Q) + 1)/2
    df = pd.DataFrame(Q,dtype=int)
    df['code'] = df.astype(str).agg(''.join, axis=1)
    df = df.reset_index()
    df = df[['index','code']]
    return df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']

def sample_pseudo_bulk(pseudobulk_map,sample_size):
    pseudobulk_map_sample = {}
    for key, value in pseudobulk_map.items():
        if len(value)>sample_size:
            pseudobulk_map_sample[key] = random.sample(value,sample_size)
        else:
            pseudobulk_map_sample[key] = value
    return pseudobulk_map_sample     

def get_pseudobulk(omtx,rp_mat,downsample_pseudobulk,downsample_size,mode,res=None):
    import anndata as an
    import scanpy as sc

    adata = an.AnnData(omtx.T)
    sc.pp.filter_cells(adata,min_counts=1e3)
    sc.pp.normalize_total(adata,exclude_highly_expressed=True,target_sum=1e6)
    mtx = adata.X.T
    
    pseudobulk_map = get_projection_map(mtx,rp_mat)

    if downsample_pseudobulk:
        pseudobulk_map = sample_pseudo_bulk(pseudobulk_map,downsample_size)

    pseudobulk = []
    for _, value in pseudobulk_map.items():
        pseudobulk.append(mtx[:,value].mean(1))

    pseudobulk = np.array(pseudobulk).T
    scaler = StandardScaler()
    pseudobulk = np.exp(scaler.fit_transform(np.log1p(pseudobulk)))


    if mode == 'full':
        return {mode:{'pb_data':pseudobulk, 'pb_map':pseudobulk_map}}
    else:
         res.put({mode:{'pb_data':pseudobulk, 'pb_map':pseudobulk_map}})
