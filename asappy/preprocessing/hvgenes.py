import logging
import numpy as np
import pandas as pd
import numba as nb
from math import sqrt
logger = logging.getLogger(__name__)


@nb.njit(parallel=True)
def calc_res(mtx,sum_gene,sum_cell,sum_total,theta,clip,n_gene,n_cell):
    
    def clac_clipped_res_dense(gene: int, cell: int) -> np.float64:
        mu = sum_gene[gene] * sum_cell[cell] / sum_total
        value = mtx[cell, gene]

        mu_sum = value - mu
        pre_res = mu_sum / sqrt(mu + mu * mu / theta)
        res = np.float64(min(max(pre_res, -clip), clip))
        return res

    norm_gene_var = np.zeros(n_gene, dtype=np.float64)

    for gene in nb.prange(n_gene):
        sum_clipped_res = np.float64(0.0)
        for cell in range(n_cell):
            sum_clipped_res += clac_clipped_res_dense(gene, cell)
        mean_clipped_res = sum_clipped_res / n_cell

        var_sum = np.float64(0.0)
        for cell in range(n_cell):
            clipped_res = clac_clipped_res_dense(gene, cell)
            diff = clipped_res - mean_clipped_res
            var_sum += diff * diff

        norm_gene_var[gene] = var_sum / n_cell

    return norm_gene_var

def select_hvgenes(mtx,method,z_high_gene_expression,z_high_gene_var):
    '''
    adapted from pyliger plus scanpy's seurat high variable gene selection

    '''
    if method == 'apearson':
        
        ### remove high expression genes with std > 10
        from scipy.stats import zscore
        data = np.array(mtx.mean(axis=0))
        z_scores = zscore(data)
        mean_genef_index = z_scores > z_high_gene_expression
        logging.info('removed high expression genes...'+str(mean_genef_index.sum()))

        ## remove genes with total sum over all cells as zero
        sum_gene = np.array(mtx.sum(axis=0))
        sum_genef_index = sum_gene!=0
        logging.info('kept non zero sum expression genes...'+str(sum_genef_index.sum()))
        
        ### combine two filters
        genef_index = np.array([a or b for a, b in zip(mean_genef_index, sum_genef_index)])
        mtx = mtx[:,genef_index]
                
        df = pd.DataFrame()
        df['gzero']=genef_index
        
        sum_gene = np.array(mtx.sum(axis=0)).ravel()
        sum_cell = np.array(mtx.sum(axis=1)).ravel()
        sum_total = np.float64(np.sum(sum_gene).ravel())
        n_gene = mtx.shape[1]
        n_cell = mtx.shape[0]
        
        theta = np.float64(100)
        clip = np.float64(np.sqrt(n_cell))
        norm_gene_var = calc_res(mtx,sum_gene,sum_cell,sum_total,theta,clip,n_gene,n_cell)
        select_genes = norm_gene_var>z_high_gene_var 
        
        logging.info('kept high variance genes...'+str(select_genes.sum()))

        prev_indx = df.loc[df['gzero']==True].index.values
        sgenemap = {x:y for x,y in zip(prev_indx,select_genes)}
        
        df['select'] = [sgenemap[x]  if x in sgenemap.keys() else False for x in df.index.values]
        
        return mtx[:,select_genes], df['select'].values

    elif method =='seurat':
        from skmisc.loess import loess
        
                
        ### remove high expression genes with std > 10
        from scipy.stats import zscore
        data = np.array(mtx.mean(axis=0))
        z_scores = zscore(data)
        mean_genef_index = z_scores > z_high_gene_expression


        ## remove genes with total sum over all cells as zero
        sum_gene = np.array(mtx.sum(axis=0))
        sum_genef_index = sum_gene!=0
        
        ### combine two filters
        genef_index = np.array([a or b for a, b in zip(mean_genef_index, sum_genef_index)])
        mtx = mtx[:,genef_index]
                
        df = pd.DataFrame()
        df['gzero']=genef_index

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
        #######
        
        clip_val_broad = np.broadcast_to(clip_val, mtx.shape)

        np.putmask(
        mtx,
        mtx > clip_val_broad,
        clip_val_broad
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
        
        select_genes = norm_gene_var> z_high_gene_var
        prev_indx = df.loc[df['gzero']==True].index.values
        sgenemap = {x:y for x,y in zip(prev_indx,select_genes)}
        
        df['select'] = [sgenemap[x]  if x in sgenemap.keys() else False for x in df.index.values]
        
        return mtx[:,select_genes], df['select'].values
    
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
        trc_per_cell = mtx.T.sum(1)

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
    
