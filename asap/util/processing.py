import os
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

import logging
logger = logging.getLogger(__name__)

def filter_minimal(df,cutoff):
	
	drop_columns = [ col for col,val  in df.sum(axis=0).iteritems() if val < cutoff ]

	logger.info('genes to filter based on mincout cutoff - '+ str(len(drop_columns)))

	for mt_g in [x for x in df.columns if 'MT-' in x]:drop_columns.append(mt_g)

	logger.info('adding mitochondrial genes - '+ str(len(drop_columns)))

	for spk_g in [x for x in df.columns if 'ERCC' in x]:drop_columns.append(spk_g)

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


def tcell_preprocessing():

	fp ='/data/Tcell/GSE156728/'
	op = '/home/BCCRC.CA/ssubedi/project_data/data/Tcell'
	samples =	[
		'GSE156728_BC_10X.CD4.counts.txt.gz',
		'GSE156728_BC_10X.CD8.counts.txt.gz',
		'GSE156728_BCL_10X.CD4.counts.txt.gz',
		'GSE156728_BCL_10X.CD8.counts.txt.gz',
		'GSE156728_ESCA_10X.CD4.counts.txt.gz',
		'GSE156728_ESCA_10X.CD8.counts.txt.gz',
		'GSE156728_MM_10X.CD4.counts.txt.gz',
		'GSE156728_MM_10X.CD8.counts.txt.gz',
		'GSE156728_PACA_10X.CD4.counts.txt.gz',
		'GSE156728_PACA_10X.CD8.counts.txt.gz',
		'GSE156728_RC_10X.CD4.counts.txt.gz',
		'GSE156728_RC_10X.CD8.counts.txt.gz',
		'GSE156728_THCA_10X.CD4.counts.txt.gz',
		'GSE156728_THCA_10X.CD8.counts.txt.gz',
		'GSE156728_UCEC_10X.CD4.counts.txt.gz',
		'GSE156728_UCEC_10X.CD8.counts.txt.gz',
		'GSM4743199_OV_10X.CD4.counts.txt.gz',
		'GSM4743199_OV_10X.CD8.counts.txt.gz',
		'GSM4743231_FTC_10X.CD4.counts.txt.gz',
		'GSM4743231_FTC_10X.CD8.counts.txt.gz',
		'GSM4743237_CHOL_SS2.CD4.counts.txt.gz',
		'GSM4743237_CHOL_SS2.CD8.counts.txt.gz'
		]
	df_combine = pd.DataFrame()
	for sample in samples:
		print("processing--"+sample)
		df = pd.read_csv(fp+sample,sep="\t").T
		df = df.rename(columns=df.iloc[0])
		df = df.iloc[1:].reset_index()
		df = df.rename(columns={"index":"cell"})
		df['sample'] = sample.split('_')[1]+"_"+sample.split('.')[1]
		
		if df.shape[0] > 2500:
			df = df.sample(n=2500)  
		
		df_combine = pd.concat([df_combine, df], axis=0, ignore_index=True)
		print(df.shape,df_combine.shape)

		del df
	
	df_combine.values[df_combine.isna()] = 0
	df_combine.to_csv(op+"tcell_counts.txt.gz",index=False,sep="\t",compression="gzip")


def lung_preprocessing():

	fp ='/data/NSLC/GSE148071/'
	op = '/home/BCCRC.CA/ssubedi/project_data/data/lung/'
	samples =[f for f in listdir(fp) if isfile(join(fp, f))]

	df_combine = pd.DataFrame()
	for sample in samples:
		print("processing--"+sample)
		df = pd.read_csv(fp+sample,sep="\t").T
		df = df.reset_index()
		df = df.rename(columns={"index":"cell"})
		df['sample'] = sample.split('_')[1]+"_"+sample.split('.')[1]

		print(df.shape)		

		if df.shape[0] > 2500:
			df = df.sample(n=2500)  
		
		df_combine = pd.concat([df_combine, df], axis=0, ignore_index=True)
		print(df.shape,df_combine.shape)

		# del df
	
	df_combine.values[df_combine.isna()] = 0
	df_combine.to_pickle(op+"lungs_counts.pkl")

lung_preprocessing()
