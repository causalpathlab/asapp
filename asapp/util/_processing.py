import os
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

def filter_minimal(df,cutoff):
	
	drop_columns = [ col for col,val  in df.sum(axis=0).iteritems() if val < cutoff ]

	logger.info('genes to filter based on mincout cutoff - '+ str(len(drop_columns)))

	for mt_g in [x for x in df.columns if 'MT-' in x]:
		drop_columns.append(mt_g)

	logger.info('adding mitochondrial genes - '+ str(len(drop_columns)))

	for spk_g in [x for x in df.columns if 'ERCC' in x]:
		drop_columns.append(spk_g)

	logger.info('adding spikes - '+ str(len(drop_columns)))

	return drop_columns

def tenx_preprocessing(fpath,sample_id):

	from scipy.io import mmread
	from os import fspath
	import scipy.sparse 
	import gc

	logger.info('reading matrix file...')
	dtype='float32'
	X = mmread(fspath(fpath+'matrix.mtx.gz')).astype(dtype)

	df = pd.DataFrame(X.todense())
	df = df.T
	cols = pd.read_csv(fpath+'genes.tsv.gz',header=None)
	df.columns = cols.values.flatten()

	del X
	gc.collect()

	non_zero = np.count_nonzero(df)
	total_val = np.product(df.shape)
	sparsity = (total_val - non_zero) / total_val
	
	logger.info(f"shape:{str(df.shape)}")
	logger.info(f"sparsity:{str(sparsity)}")
	logger.info(f"gene:{str(df.sum(0).max())} ,{str(df.sum(0).min())}")
	logger.info(f"cell: {str(df.sum(1).max())} , {str(df.sum(1).min())}")

	min_total_gene_count = 100
	drop_columns = filter_minimal(df,min_total_gene_count)
	df = df.drop(drop_columns,axis=1)
	logger.info(f"shape after filter:{str(df.shape)}")
	
	## generate npz files
	logger.info('processing--creating filtered npz files')

	smat = scipy.sparse.csr_matrix(df.to_numpy())
	np.save(fpath+sample_id+'.indptr',smat.indptr)
	np.save(fpath+sample_id+'.indices',smat.indices)
	np.save(fpath+sample_id+'.data',smat.data)
	pd.Series(df.columns).to_csv(fpath+sample_id+'_genes.txt.gz',index=False,header=None)
	logger.info('Data pre-processing--COMPLETED !!')
