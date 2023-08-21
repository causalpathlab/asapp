import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix
import h5py as hf
import tables
import glob
import os

from ..dutil import DataSet, CreateDatasetFromH5, CreateDatasetFromMTX, CreateDatasetFromH5AD, data_fileformat
from ..asappy import asap

import logging
logger = logging.getLogger(__name__)

def create_asap(sample,data_size,number_batches=1):
	
	"""        
    Attributes:
		filename(str):
			The filename to store asap object.
        data_size(int):
			The total number of cells to analyze either from one data file or multiple files.
        number_batches(int):
			The total number of batches to use for analysis, each batch will have data_size cells.
	"""

	filetype = data_fileformat()
	## read source files and create dataset for asap
	if filetype == 'h5':
		ds = CreateDatasetFromH5('./data/') 
		print(ds.peek_datasets())
		ds.create_asapdata(sample) 
	elif filetype == 'h5ad':
		ds = CreateDatasetFromH5AD('./data/') 
		print(ds.peek_datasets())
		ds.create_asapdata(sample) 

	## create anndata like object for asap 
	adata = DataSet(sample)
	dataset_list = adata.get_dataset_names()
	adata.initialize_data(dataset_list=dataset_list,batch_size=data_size)
	return asap(adata,number_batches)

	