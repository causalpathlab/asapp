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

def sim_data(sca,N,K,P):
	import _sim
	H = _sim.generate_H(N, K)
	W = _sim.generate_W(P, K)
	R = np.matmul(H.T, W.T) 
	X = np.random.poisson(R)

	sca.data.rows = ['c_'+str(i) for i in range(N) ]
	sca.data.cols = ['g_'+str(i) for i in range(P) ]
	sca.data.mtx = np.asmatrix(X)

	return H,W

