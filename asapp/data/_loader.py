import pandas as  pd
import numpy as np
import logging
from scipy.sparse import csr_matrix
logger = logging.getLogger(__name__)


def initialize_data(sca):
	
	fpath = sca.config.home + sca.config.experiment + sca.config.input + sca.config.sample_id + '/' + sca.config.sample_id
	
	sca.data.mtx_indptr = fpath + sca.config.mtx_indptr
	sca.data.mtx_indices = fpath + sca.config.mtx_indices
	sca.data.mtx_data = fpath + sca.config.mtx_data
	
	sca.data.rows = list(pd.read_csv(fpath + sca.config.rows)['rows']) 
	sca.data.cols = list(pd.read_csv(fpath + sca.config.cols)['cols']) 

def load_data(sca):

	mtx_indptr = np.load(sca.data.mtx_indptr)
	mtx_indices = np.load(sca.data.mtx_indices)
	mtx_data = np.load(sca.data.mtx_data)

	mtx_dim = len(sca.data.cols)
	row_ids = sca.data.rows

	rows = csr_matrix((mtx_data,mtx_indices,mtx_indptr),shape=(len(row_ids),mtx_dim))
	
	sca.data.mtx= rows.todense()