import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix
import h5py as hf
import tables
import glob
import os

from ..dutil import DataSet, CreateDatasetFromH5, CreateDatasetFromMTX, CreateDatasetFromH5AD, data_fileformat
from ..asappy import asap
from ..util.logging import setlogger

import logging
logger = logging.getLogger(__name__)

def create_asap_data(sample,working_dirpath,select_genes=None):
	
	"""        
    Attributes:
		filename(str):
			The filename to store asap object.
        data_size(int):
			The total number of cells to analyze either from one data file or multiple files.
        number_batches(int):
			The total number of batches to use for analysis, each batch will have data_size cells.
	"""

	setlogger(sample=sample,sample_dir=working_dirpath)
	
	number_of_selected_genes = 0
	if isinstance(select_genes,list):
		number_of_selected_genes = len(select_genes)

	logging.info('Creating asap data... \n'+
	'sample :' + str(sample)+'\n'+
	'number_of_selected_genes :' + str(number_of_selected_genes)+'\n'
	)

	filetype = data_fileformat(sample,working_dirpath)
	## read source files and create dataset for asap
	if filetype == 'h5':
		ds = CreateDatasetFromH5(working_dirpath,sample) 
		print(ds.peek_datasets())
		ds.create_asapdata(sample,select_genes) 
	elif filetype == 'h5ad':
		ds = CreateDatasetFromH5AD(working_dirpath,sample) 
		print(ds.peek_datasets())
		ds.create_asapdata(sample,select_genes) 
	
	logging.info('Completed asap data.')

def create_asap_object(sample,data_size,working_dirpath,number_batches=1):

	setlogger(sample,working_dirpath)

	logging.info('Creating asap object... \n'+
		'data_size :' + str(data_size)+'\n'+
		'number_batches :' + str(number_batches)+'\n'
		)

	## create anndata like object for asap 
	adata = DataSet(sample,number_batches,working_dirpath)
	dataset_list = adata.get_dataset_names()
	adata.initialize_data(dataset_list=dataset_list,batch_size=data_size)
	return asap(adata)

	