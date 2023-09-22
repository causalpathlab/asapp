import logging
import numpy as np
logger = logging.getLogger(__name__)

from sklearn.preprocessing import normalize, StandardScaler

def normalization_raw(mtx,method):

    from sklearn.preprocessing import normalize

    if method == 'unitnorm':
        norm_data = normalize(mtx, norm='l1')
        return norm_data.T 
    
    elif method =='fscale':
        gene_sums = mtx.sum(axis=0)
        target_sum = np.median(gene_sums)
        # target_sum = 1e4
        gene_sums = gene_sums/target_sum
        mtx_norm = np.divide(mtx,gene_sums)
        return mtx_norm.T
    
    elif method =='scpy':
        import anndata as an
        import scanpy as sc

        adata = an.AnnData(mtx)
        # sc.pp.normalize_total(adata,exclude_highly_expressed=True,target_sum=1e4)
        sc.pp.normalize_total(adata)
        return adata.X.T
    
def normalization_pb(mtx,method):
    
    if method == 'fscale':
        gene_sums = mtx.sum(axis=0)
        target_sum = np.median(gene_sums)
        # target_sum = 1e4
        gene_sums = gene_sums/target_sum
        mtx_norm = np.divide(mtx,gene_sums)
        return mtx_norm.T

    elif method == 'rscale':
        gene_rms = np.sqrt(np.mean(mtx**2, axis=0))
        mtx_norm = mtx / gene_rms
        return mtx_norm.T
    
    elif method == 'lscale':
        scaler = StandardScaler()
        return np.exp(scaler.fit_transform(np.log1p(mtx.T)))

def preprocess_pb(mtx,gene_mean_z):
    
    ### remove high expression genes with std > 10
    from scipy.stats import zscore
    data = np.array(mtx.mean(axis=0))
    z_scores = zscore(data)
    mean_genef_index = z_scores > gene_mean_z
    logging.info('removed high expression genes...'+str(mean_genef_index.sum()))

    ## remove genes with total sum over all cells as zero
    sum_gene = np.array(mtx.sum(axis=0))
    sum_genef_index = sum_gene!=0
    logging.info('kept non zero sum expression genes...'+str(sum_genef_index.sum()))
    
    ### combine two filters
    genef_index = np.array([a or b for a, b in zip(mean_genef_index, sum_genef_index)])
    return mtx[:,genef_index], genef_index
    
