import numpy as np
import pandas as pd
import logging
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from ..preprocessing.normalize import normalization_pb,normalization_raw, preprocess_pb
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

<<<<<<< HEAD
def get_pseudobulk(mtx,rp_mat,downsample_pseudobulk,downsample_size,mode,normalize_raw=None, normalize_pb=None,hvg_selection=False,gene_mean_z=None,gene_var_z=None,res=None):  
     
    if normalize_raw is not None:
        logging.info('normalize raw data -'+normalize_raw)    
        mtx = normalization_raw(mtx.T,normalize_raw)
 
=======
def get_pseudobulk(mtx,rp_mat,downsample_pseudobulk,downsample_size,mode,normalization_raw, normalization_pb,res=None):   

    logging.info('normalize raw data -'+normalization_raw)    
    mtx = normalize_raw(mtx,normalization_raw)
    
>>>>>>> parent of cfd89de... gene selection - working version
    pseudobulk_map = get_projection_map(mtx,rp_mat)

    if downsample_pseudobulk:
        pseudobulk_map = sample_pseudo_bulk(pseudobulk_map,downsample_size)
<<<<<<< HEAD
        
=======

    logging.info('normalize raw data for aggregation -'+'lognorm')    
    mtx_norm = normalize_raw(mtx,method='lognorm')

    mtx_norm = mtx
>>>>>>> parent of cfd89de... gene selection - working version
    pseudobulk = []
    for _, value in pseudobulk_map.items():
        # m = mtx_norm[:,value]    
        # lambda_estimates = np.mean(m, axis=1)
        # s = poisson.rvs(mu=lambda_estimates, size=m.shape[0])
        # pseudobulk.append(s)

        pseudobulk.append(mtx_norm[:,value].sum(1))
        
<<<<<<< HEAD
    pseudobulk = np.array(pseudobulk).astype(np.float64)
=======
    pseudobulk = np.array(pseudobulk)

    print(pseudobulk.shape)
    hvg = 'seurat'
    logging.info('select highly variable genes pb data -'+hvg)    
    pseudobulk,hvgenes = select_hvgenes(pseudobulk,method=hvg)
    print(pseudobulk.shape)
    logging.info('normalize pb data -'+normalization_pb)    
    pseudobulk = normalize_pb(pseudobulk,normalization_pb)
    print(pseudobulk.shape)

    print('pbsum...')
    print(pseudobulk.sum())
    
    pseudobulk = pseudobulk.T 
>>>>>>> parent of cfd89de... gene selection - working version
        
    logging.info('before pseudobulk preprocessing-'+str(pseudobulk.shape)) 
    pseudobulk,gene_filter_index = preprocess_pb(pseudobulk,gene_mean_z)
    logging.info('after pseudobulk preprocessing-'+str(pseudobulk.shape)) 

    if hvg_selection:
        logging.info('before high variable genes selection...')    
        pseudobulk,hvgenes = select_hvgenes(pseudobulk,gene_filter_index,gene_var_z)
        logging.info('after high variance genes selection...'+str(pseudobulk.shape))
    else:
        logging.info('no high variance gene selection-'+str(pseudobulk.shape)) 
        hvgenes = gene_filter_index
    
    if normalize_pb != None:
        logging.info('normalize pb data -'+normalize_pb)
        pseudobulk = normalization_pb(pseudobulk,normalize_pb)
        logging.info('pseudobulk shape...'+str(pseudobulk.shape))
        logging.info('pseudobulk data...\n min '+str(pseudobulk.min())+'\n max '+str(pseudobulk.max())+'\n sum '+str(pseudobulk.sum()))

    pseudobulk = pseudobulk.astype(int)
            
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
