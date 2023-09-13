import logging
import numpy as np
logger = logging.getLogger(__name__)

def normalize_raw(mtx,method,sf=None):

    from sklearn.preprocessing import normalize

    if method == 'unitnorm':
        # Sample(row)-wise normalization
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

def normalize_pb(mtx,method,sf=None):

    if method=='lognorm':
        row_sums = mtx.sum(axis=0)
        if sf == None:
            target_sum = np.median(row_sums)
        else :
            target_sum = sf
        row_sums = row_sums/target_sum
        mtx_norm = np.divide(mtx,row_sums)
        return np.log1p(mtx_norm)
    
    elif method =='robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler(with_centering=False,unit_variance=True)
        mtx_norm = scaler.fit_transform(mtx.T)
        return mtx_norm.T

    elif method =='scaler':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=False)
        mtx_norm = np.exp(scaler.fit_transform(np.log1p(mtx.T)))
        # mtx_norm = scaler.fit_transform(mtx.T)
        return mtx_norm.T

    elif method =='rtmsqr':
        gene_sum_sq = np.square(mtx).sum(axis=0)
        scaler = 1 / np.sqrt(gene_sum_sq / (mtx.shape[0] - 1))
        mtx_norm = mtx/scaler
        return mtx_norm