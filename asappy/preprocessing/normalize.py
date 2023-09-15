import logging
import numpy as np
logger = logging.getLogger(__name__)

from sklearn.preprocessing import normalize

def normalize_raw(mtx,method,sf=None):

    if method == 'unitnorm':
        norm_data = normalize(mtx.T, norm='l1')
        return norm_data.T 
    
    elif method =='lognorm':
        row_sums = mtx.T.sum(axis=1)
        if sf == None:
            target_sum = np.median(row_sums)
        else :
            target_sum = sf
        row_sums = row_sums/target_sum
        mtx_norm = np.divide(mtx,row_sums)
        return np.log1p(mtx_norm)

def normalize_pb(mtx,method):
    if method == 'mscale':
        row_sums = mtx.T.sum(axis=1)
        target_sum = np.median(row_sums)
        row_sums = row_sums/target_sum
        mtx_norm = np.divide(mtx,row_sums)
        return mtx_norm

    elif method == 'rscale':
        gene_rms = np.sqrt(np.mean(mtx**2, axis=0))
        return mtx / gene_rms

