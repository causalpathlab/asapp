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

def adjust_rp_weight(mtx,rp_mat_list,weight='mean',hvg_percentile=99):

    if weight == 'std':
        gene_w = np.std(mtx,axis=1)
    elif weight == 'mean':
        gene_w = np.mean(mtx,axis=1)
    elif weight == 'hvg':
        genes_var = get_gene_norm_var(mtx.T)
        cutoff_percentile = np.percentile(genes_var, hvg_percentile)
        print(cutoff_percentile,genes_var.min(),genes_var.mean(),genes_var.max())
        genes_var_sel = np.where(genes_var < cutoff_percentile, 0, genes_var)    
        gene_w = np.mean(mtx,axis=1)
        print((genes_var_sel !=0).sum())
        gene_w = np.where(genes_var_sel == 0, gene_w,np.exp(gene_w))
    
    rp_mat_w_list = []
    for rp_mat in rp_mat_list:    
        rp_mat_w_list.append(rp_mat * gene_w)
    rp_mat_w = np.mean(rp_mat_w_list, axis=0)
    
    return rp_mat_w

def get_random_projection_data(mtx,rp_mat_list):
    rp_mat_w = adjust_rp_weight(mtx,rp_mat_list)
    return np.dot(rp_mat_w,mtx).T

def get_projection_map(mtx,rp_mat_list,min_pseudobulk_size=50):
    
    rp_mat_w = adjust_rp_weight(mtx,rp_mat_list)
    Q = np.dot(rp_mat_w,mtx)
    
    ## center for PCA
    scaler = StandardScaler(with_std=False)
    Q = scaler.fit_transform(Q.T)
    pca = PCA(n_components=Q.shape[1])
    Z = pca.fit_transform(Q)

    #### binarization method
    # Z = (np.sign(Z) + 1)/2
    # df = pd.DataFrame(Z,dtype=int)
    # df['code'] = df.astype(str).agg(''.join, axis=1)
    # df = df.reset_index()
    # df = df[['index','code']]
    # return df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']
    
    
    ## quantization method
    # num_qlevels = 100
    # min_value,max_value = Z.min(),Z.max()
    # bin_width = (max_value - min_value) / num_qlevels
    # df = pd.DataFrame(Z)
    # Z = np.digitize(Z, np.arange(min_value, max_value, bin_width))
    # df['code'] = df.astype(str).agg(''.join, axis=1)
    # df = df.reset_index()
    # df = df[['index','code']]
    # print(df['code'].nunique())
    # return df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index'] 
    
    n_clust = min(max(Q.shape[1]/100,min_pseudobulk_size),1000)
    kmeans = KMeans(n_clusters=n_clust, random_state=0)
    kmeans.fit(Z)
    cluster_labels = kmeans.labels_ 
    df = pd.DataFrame()
    bin_code = [str(number) for number in cluster_labels]
    df['code'] = bin_code 
    print(df['code'].nunique())
    df = df.reset_index()
    return df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']
    
    # from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    # distance_threshold = 250
    # linked = linkage(q, method='ward')
    # cluster_labels = fcluster(linked,  t=distance_threshold, criterion='distance')

    # df = pd.DataFrame()
    # bin_code = [format(number, '010b') for number in cluster_labels]
    # df['code'] = bin_code 
    # print(df.code.nunique())
    # df = df.reset_index()
    # return df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']


def sample_pseudo_bulk(pseudobulk_map,sample_size):
    pseudobulk_map_sample = {}
    for key, value in pseudobulk_map.items():
        if len(value)>sample_size:
            pseudobulk_map_sample[key] = random.sample(value,sample_size)
        else:
            pseudobulk_map_sample[key] = value
    return pseudobulk_map_sample     

def get_pseudobulk(mtx,rp_mat,min_pseudobulk_size,downsample_pseudobulk,downsample_size,mode,normalize_raw=None, normalize_pb=None,hvg_selection=False,gene_mean_z=None,gene_var_z=None,res=None):  
     
    if normalize_raw is not None:
        logging.info('normalize raw data -'+normalize_raw)    
        mtx = normalization_raw(mtx.T,normalize_raw)
 
    pseudobulk_map = get_projection_map(mtx,rp_mat,min_pseudobulk_size)

    if downsample_pseudobulk:
        pseudobulk_map = sample_pseudo_bulk(pseudobulk_map,downsample_size)
        
    pseudobulk = []
    for _, value in pseudobulk_map.items():
        m = mtx[:,value]    
        lambda_estimates = np.mean(m, axis=1)
        s = poisson.rvs(mu=lambda_estimates, size=m.shape[0])
        pseudobulk.append(s)
        # pseudobulk.append(mtx[:,value].sum(1))
        
    pseudobulk = np.array(pseudobulk).astype(np.float64)
        
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

def get_randomprojection(mtx,rp_mat_list,mode,normalization,res=None):   

    rp_mat = get_random_projection_data(mtx,rp_mat_list)

    if mode == 'full':
        return {mode:rp_mat}
    else:
         res.put({mode:{'rp_data':rp_mat}})
