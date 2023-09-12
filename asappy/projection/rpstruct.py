import numpy as np
import pandas as pd
import logging
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from ..preprocessing.normalize import normalize_pb,normalize_raw
from ..preprocessing.hvgenes import select_hvgenes
from scipy.stats import poisson
import random
logger = logging.getLogger(__name__)

def projection_data(depth,ndims):
    rp = []
    for _ in range(depth):
        rp.append(np.random.normal(size = (ndims,1)).flatten())                      
    return np.asarray(rp)
	

def get_random_projection_data(mtx,rp_mat):

    Z = np.dot(rp_mat,mtx)
    _, _, Q = randomized_svd(Z, n_components= Z.shape[0], random_state=0)
    
    scaler = StandardScaler()
    Q = scaler.fit_transform(Q.T)
    
    return Q

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

def get_pseudobulk(mtx,rp_mat,downsample_pseudobulk,downsample_size,mode,normalization_raw, normalization_pb,res=None):   

    logging.info('normalize raw data -'+normalization_raw)    
    mtx = normalize_raw(mtx,normalization_raw)
    
    pseudobulk_map = get_projection_map(mtx,rp_mat)

    if downsample_pseudobulk:
        pseudobulk_map = sample_pseudo_bulk(pseudobulk_map,downsample_size)

    logging.info('normalize raw data for aggregation -'+'lognorm')    
    mtx_norm = normalize_raw(mtx,method='lognorm')
    pseudobulk = []
    for _, value in pseudobulk_map.items():
        # m = mtx_norm[:,value]    
        # lambda_estimates = np.mean(m, axis=1)
        # s = poisson.rvs(mu=lambda_estimates, size=m.shape[0])
        # pseudobulk.append(s)

        pseudobulk.append(mtx_norm[:,value].sum(1))
        
    pseudobulk = np.array(pseudobulk).T

    print(pseudobulk.shape)
    logging.info('select highly variable genes pb data -seurat')    
    pseudobulk,hvgenes = select_hvgenes(pseudobulk.T,method='seurat')
    print(pseudobulk.shape)
    logging.info('normalize pb data -'+normalization_pb)    
    pseudobulk = normalize_pb(pseudobulk,normalization_pb)
    print(pseudobulk.shape)

    print('pbsum...')
    print(pseudobulk.sum())
    
    if mode == 'full':
        return {mode:{'pb_data':pseudobulk, 'pb_map':pseudobulk_map,'pb_hvgs':hvgenes}}
    else:
         res.put({mode:{'pb_data':pseudobulk, 'pb_map':pseudobulk_map, 'pb_hvgs':hvgenes}})

def get_randomprojection(mtx,rp_mat,mode,normalization,res=None):   

    rp_mat = get_random_projection_data(mtx,rp_mat)

    if mode == 'full':
        return {mode:rp_mat}
    else:
         res.put({mode:{'rp_data':rp_mat}})
