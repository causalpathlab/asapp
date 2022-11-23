import pandas as  pd
import numpy as np
import logging
from scipy.sparse import csr_matrix
logger = logging.getLogger(__name__)


def initialize_data(pine):
	
	fpath = pine.config.home + pine.config.experiment + pine.config.input + pine.config.sample_id + '/' + pine.config.sample_id
	
	pine.data.mtx_indptr = fpath + pine.config.mtx_indptr
	pine.data.mtx_indices = fpath + pine.config.mtx_indices
	pine.data.mtx_data = fpath + pine.config.mtx_data
	
	pine.data.rows = list(pd.read_csv(fpath + pine.config.rows)['rows']) 
	pine.data.cols = list(pd.read_csv(fpath + pine.config.cols)['cols']) 

def load_data(pine):

	mtx_indptr = np.load(pine.data.mtx_indptr)
	mtx_indices = np.load(pine.data.mtx_indices)
	mtx_data = np.load(pine.data.mtx_data)

	mtx_dim = len(pine.data.cols)
	row_ids = pine.data.rows

	rows = csr_matrix((mtx_data,mtx_indices,mtx_indptr),shape=(len(row_ids),mtx_dim))
	
	pine.data.mtx= rows.todense()