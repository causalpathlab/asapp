import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix

class DataSet:
		def __init__(self):
			self.config = None
			self.mtx_indptr = None
			self.mtx_indices = None
			self.mtx_data = None
			self.rows = None
			self.cols = None
			self.mtx = None

		def initialize_path(self):
			
			self.inpath = self.config.home + self.config.experiment + self.config.input + self.config.sample_id + '/' + self.config.sample_id
			self.outpath = self.config.home + self.config.experiment + self.config.output + self.config.sample_id + '/' + self.config.sample_id

		def initialize_data(self):

			self.mtx_indptr = self.inpath + self.config.mtx_indptr
			self.mtx_indices = self.inpath + self.config.mtx_indices
			self.mtx_data = self.inpath + self.config.mtx_data
			
			self.rows = list(pd.read_csv(self.inpath + self.config.rows)['rows']) 
			self.cols = list(pd.read_csv(self.inpath + self.config.cols)['cols']) 

		def load_data(self):

			mtx_indptr = np.load(self.mtx_indptr)
			mtx_indices = np.load(self.mtx_indices)
			mtx_data = np.load(self.mtx_data)

			mtx_dim = len(self.cols)
			row_ids = self.rows

			rows = csr_matrix((mtx_data,mtx_indices,mtx_indptr),shape=(len(row_ids),mtx_dim))
			
			self.mtx= rows.todense()

		def sim_data(self,N,K,P):
			from util import _sim as _sim
			H = _sim.generate_H(N, K)
			W = _sim.generate_W(P, K)
			R = np.matmul(H.T, W.T) 
			X = np.random.poisson(R)

			self.rows = ['c_'+str(i) for i in range(N) ]
			self.cols = ['g_'+str(i) for i in range(P) ]
			self.mtx = np.asmatrix(X)

			return H,W

