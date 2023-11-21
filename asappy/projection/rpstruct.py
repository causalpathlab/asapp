import numpy as np
import pandas as pd
import logging
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,QuantileTransformer

from sklearn.cluster import KMeans

from ..preprocessing.normalize import normalization_pb,normalization_raw, preprocess_pb
from ..preprocessing.hvgenes import select_hvgenes,get_gene_norm_var
from scipy.stats import poisson
import random
logger = logging.getLogger(__name__)

def projection_data(depth,ndims,nsample=10):
    rp_list = []
    for iter_o in range(nsample):
        rp = []
        np.random.seed(iter_o)
        for iter_i in range(depth):
            rp.append(np.random.normal(size = (ndims,1)).flatten())                      
        rp_list.append(np.asarray(rp))
    return rp_list

def adjust_rp_weight(mtx,rp_mat_list,weight='std',hvg_percentile=99):

    if weight == 'std':
        gene_w = np.std(mtx,axis=1)
    elif weight == 'mean':
        gene_w = np.mean(mtx,axis=1)
    elif weight == 'mean_whvg':
        genes_var = get_gene_norm_var(mtx.T)
        cutoff_percentile = np.percentile(genes_var, hvg_percentile)
        print(cutoff_percentile,genes_var.min(),genes_var.mean(),genes_var.max())
        genes_var_sel = np.where(genes_var < cutoff_percentile, 0, genes_var)    
        gene_w = np.mean(mtx,axis=1)
        print((genes_var_sel !=0).sum())
        gene_w = np.where(genes_var_sel == 0, gene_w,np.exp(gene_w))
    
    rp_mat_w_list = []
    for rp_mat in rp_mat_list:rp_mat_w_list.append(rp_mat * gene_w)
    
    rp_mat_w = rp_mat_w_list[0]
    for rp_mat in rp_mat_w_list[1:]:rp_mat_w = np.vstack((rp_mat_w,rp_mat))
    
    return rp_mat_w

def get_random_projection_data(mtx,rp_mat_list):
    rp_mat_w = adjust_rp_weight(mtx,rp_mat_list)
    Q = np.dot(rp_mat_w,mtx).T
    pca = PCA(n_components=rp_mat_list[0].shape[0])
    Z = pca.fit_transform(Q)
    return Z            


def get_projection_map(mtx,rp_mat_list,method='SVD'):

    rp_mat_w = adjust_rp_weight(mtx,rp_mat_list)

    logging.info('pb aggregation -'+method)  
    
    if method =='PCA':
        Q = np.dot(rp_mat_w,mtx).T
        pca = PCA(n_components=rp_mat_list[0].shape[0])    
        Z = pca.fit_transform(Q)
        scaler = StandardScaler()
        Z = scaler.fit_transform(Z)
    
    if method =='SVD':
        Q = np.dot(rp_mat_w,mtx)
        _, _, Z = randomized_svd(Q, n_components= 10, random_state=0)
        scaler = StandardScaler()
        Z = scaler.fit_transform(Z.T)
    
    Z = (np.sign(Z) + 1)/2
    df = pd.DataFrame(Z,dtype=int)
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

def get_pseudobulk(mtx,rp_mat,downsample_pseudobulk,downsample_size,pb_aggregation,mode,normalize_raw, normalize_pb,hvg_selection,gene_mean_z,gene_var_z,res=None):  
     
    if normalize_raw is not None:
        logging.info('normalize raw data -'+normalize_raw)    
        mtx = normalization_raw(mtx.T,normalize_raw)
 
    pseudobulk_map = get_projection_map(mtx,rp_mat,pb_aggregation)

    if downsample_pseudobulk:
        pseudobulk_map = sample_pseudo_bulk(pseudobulk_map,downsample_size)
        
    pseudobulk = []
    pseudobulk_depth = 1e4
    for _, value in pseudobulk_map.items():        
        pb = mtx[:,value].sum(1)
        pb = (pb/pb.sum()) * pseudobulk_depth
        pseudobulk.append(pb)
        
    pseudobulk = np.array(pseudobulk).astype(np.float64)
        
    logging.info('before pseudobulk preprocessing-'+str(pseudobulk.shape)) 
    gene_filter_index = preprocess_pb(pseudobulk,gene_mean_z)
    logging.info('after pseudobulk preprocessing-'+str(gene_filter_index.sum())) 

    if hvg_selection:
        logging.info('before high variable genes selection...')    
        hvgenes = select_hvgenes(pseudobulk,gene_filter_index,gene_var_z)
        logging.info('after high variable genes selection-'+str(hvgenes.sum()))
    else:
        logging.info('no high variance gene selection-'+str(pseudobulk.shape)) 
        hvgenes = gene_filter_index
    
    pseudobulk = pseudobulk[:,hvgenes]
    logging.info('after preprocessing and high variance genes selection...'+str(pseudobulk.shape))
    
    if normalize_pb != None:
        logging.info('normalize pb data -'+normalize_pb)
        pseudobulk = normalization_pb(pseudobulk,normalize_pb)
        logging.info('pseudobulk shape...'+str(pseudobulk.shape))
        logging.info('pseudobulk data...\n min '+str(pseudobulk.min())+'\n max '+str(pseudobulk.max())+'\n sum '+str(pseudobulk.sum()))
    else:
        logging.info('normalize pb data - None')
        pseudobulk = pseudobulk.T
        logging.info('pseudobulk shape...'+str(pseudobulk.shape))
        logging.info('pseudobulk data...\n min '+str(pseudobulk.min())+'\n max '+str(pseudobulk.max())+'\n sum '+str(pseudobulk.sum()))
        
    pseudobulk = pseudobulk.astype(int)
            
    if mode == 'full':
        return {mode:{'pb_data':pseudobulk, 'pb_map':pseudobulk_map,'pb_hvgs':hvgenes}}
    else:
         res.put({mode:{'pb_data':pseudobulk, 'pb_map':pseudobulk_map, 'pb_hvgs':hvgenes}})

def get_randomprojection(mtx,rp_mat_list,mode,res=None):   

    rp_mat = get_random_projection_data(mtx,rp_mat_list)

    if mode == 'full':
        return {mode:rp_mat}
    else:
         res.put({mode:{'rp_data':rp_mat}})
