import logging
import numpy as np
logger = logging.getLogger(__name__)

def select_hvgenes(mtx,method):
    
    if method =='seurat':
        '''
        adapted from scanpy seurat flavor high variable gene selection
        '''
        from skmisc.loess import loess
        
        
        gene_sum = np.sum(mtx, axis=0)
        gene_mean = gene_sum / mtx.shape[0]

        gene_mean_sq = np.multiply(mtx, mtx).mean(axis=0, dtype=np.float64)
        gene_var = gene_mean_sq - gene_mean**2
        gene_var *= mtx.shape[0] / (mtx.shape[0] - 1)

        
        not_const = gene_var > 0
        estimated_gene_var = np.zeros(mtx.shape[1], dtype=np.float64)

        y = np.log10(gene_var[not_const])
        x = np.log10(gene_mean[not_const])
        
        model = loess(x, y, span=0.3, degree=2)
        model.fit()
        estimated_gene_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10**estimated_gene_var) ## get out of log 10 space

        
        '''
        # clipped high var 
        
        stdev * sqrt(N) + mean
        '''
        #####
        N = mtx.shape[0]
        vmax = np.sqrt(N)
        clip_val = reg_std * vmax + gene_mean
        std_cuttoff = 1
        #######
        
        clip_val_broad = np.broadcast_to(clip_val, mtx.shape)

        np.putmask(
        mtx,
        mtx > clip_val_broad,
        clip_val_broad,
        )
    
        '''
        # normalized var
        
        '''
        squared_mtx_sum = np.square(mtx).sum(axis=0)
        norm_gene_var = (1 / ((N - 1) * np.square(reg_std))) * (
            (N * np.square(gene_mean))
            + squared_mtx_sum
            - 2 * gene_sum * gene_mean
        )
        select_genes = norm_gene_var>std_cuttoff
        return mtx[:,select_genes].T, select_genes
    
    elif method =='liger':
        from sklearn.preprocessing import normalize,StandardScaler
        from scipy.optimize import minimize
        import numexpr as ne
        from scipy.stats import norm

        row_sums = mtx.T.sum(axis=1)
        norm_data = (mtx / row_sums).T * 10000

        ## gene mean
        norm_sum = np.sum(norm_data, axis=0)
        norm_mean = norm_sum / norm_data.shape[0]

        ## gene var
        num_s = norm_data.shape[0]
        gene_var = norm_data.var(axis=0)
        gene_expr_var = ne.evaluate(
        "gene_var * num_s / (num_s-1)"
        )  

        alpha_thresh = 0.01
        var_thresh = 0.99

        ## total raw count per cell
        trc_per_cell = mtx_dn.T.sum(1)

        #### get lower and upper gene expr threshold
        nolan_constant = np.mean(ne.evaluate("1 / trc_per_cell"))
        alpha_thresh_corrected = alpha_thresh / norm_data.shape[1]
        gene_mean_upper = norm_mean + norm.ppf(
            1 - alpha_thresh_corrected / 2
        ) * np.sqrt(ne.evaluate("norm_mean * nolan_constant") / norm_data.shape[0])
        gene_mean_lower = ne.evaluate("log10(norm_mean * nolan_constant)")


        ## select genes
        select_gene = ne.evaluate(
            "((gene_expr_var / nolan_constant) > gene_mean_upper) & (log10(gene_expr_var) > (gene_mean_lower + var_thresh))"
        )

        print('selected genes--')
        print(select_gene.sum())

        ## mask non variable genes
        norm_data[:, ~select_gene] = 0

        ##scale
        norm_data_hvg = norm_data[:,select_gene]
        norm_sum_sq = np.sum(np.power(norm_data_hvg,2), axis=0)
        scaler = 1 / np.sqrt(norm_sum_sq / (norm_data_hvg.shape[0] - 1))
        norm_data_hvg = norm_data_hvg/scaler
        norm_data[:,:] = 0
        norm_data[:,select_gene] = norm_data_hvg


        return norm_data.T
    
