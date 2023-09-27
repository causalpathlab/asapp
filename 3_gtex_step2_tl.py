# ######################################################
# ##### bulk setup
# ######################################################

bk_sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'


######################################################
##### transfer learning
######################################################

import asappy
import anndata as an
import pandas as pd
import numpy as np

### read single cell nmf
asap_adata = an.read_h5ad('./results/'+sc_sample+'.h5asapad')
sc_beta = pd.DataFrame(asap_adata.uns['pseudobulk']['pb_beta'].T)
sc_beta.columns = asap_adata.var.index.values

## read bulk
dfbulk = pd.read_csv(outpath+'bulk.csv.gz')
dfbulk.set_index('Unnamed: 0',inplace=True)
dfbulk = dfbulk.astype(float)
bulk_sample_ids = dfbulk.index.values


## select common genes 
common_genes = [ x for x in sc_beta.columns if x in dfbulk.columns.values]
sc_beta = sc_beta.loc[:,common_genes]
dfbulk = dfbulk.loc[:,common_genes]


## normalize bulk
row_sums = dfbulk.sum(axis=1)
target_sum = np.median(row_sums)
row_sums = row_sums/target_sum
dfbulk = dfbulk.to_numpy()/row_sums[:, np.newaxis]
    
####correlation

import asapc

beta_log_scaled = asap_adata.uns['pseudobulk']['pb_beta_log_scaled'] 
pred_model = asapc.ASAPaltNMFPredict(dfbulk.T,beta_log_scaled)
pred = pred_model.predict()

bulk_corr = pd.DataFrame(pred.corr)
bulk_corr.index = bulk_sample_ids
bulk_theta = pd.DataFrame(pred.theta)
bulk_theta.index = bulk_sample_ids
bulk_corr.to_csv(outpath+'mix_bulk_corr_asap.csv.gz',compression='gzip')
bulk_theta.to_csv(outpath+'mix_bulk_theta_asap.csv.gz',compression='gzip')

#######lmfit     
# import rpy2.robjects as ro
# import rpy2.robjects.packages as rp
# import rpy2.robjects.numpy2ri
# rpy2.robjects.numpy2ri.activate()

# ro.packages.importr('limma')

# nr,nc = dfbulk.shape
# ro.r.assign("bulkdata", ro.r.matrix(dfbulk, nrow=nr, ncol=nc))

# scnr,scnc = sc_beta.T.shape
# ro.r.assign("scbeta", ro.r.matrix(sc_beta.T.to_numpy(), nrow=scnr, ncol=scnc))

# ro.r('fit <- lmFit(bulkdata,scbeta)')

# bulk_theta = pd.DataFrame(ro.r('coef(fit)'))
# bulk_theta.index = bulk_sample_ids
# bulk_theta.to_csv(outpath+'bulk_theta_lmfit.csv.gz',compression='gzip')



