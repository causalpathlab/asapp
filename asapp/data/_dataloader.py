import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
import scipy.io as sp_io
import tables

class DataSet:
		'''
		On memory - format is csr for efficiency since number of rows will be greater then cols
		On disk - h5 format, requires name and number of features in all groups are same
		'''
		def __init__(self,data_ondisk):
			self.config = None
			self.mtx_indptr = None
			self.mtx_indices = None
			self.mtx_data = None
			self.rows = None
			self.cols = None
			self.mtx = None
			self.ondisk = data_ondisk

		def initialize_path(self):
			
			self.inpath = self.config.home + self.config.experiment + self.config.input + self.config.sample_id + '/' + self.config.sample_id
			self.outpath = self.config.home + self.config.experiment + self.config.output + self.config.sample_id + '/' + self.config.sample_id

			if self.ondisk:
				self.diskfile = self.inpath+'.h5'


		def initialize_data(self):

			if self.ondisk:
				self.get_ondisk_features()
			else:
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


		def get_ondisk_features(self):
			with tables.open_file(self.diskfile, 'r') as f:
				for group in f.walk_groups():
					if '/' not in group._v_name:
						self.cols = [x.decode('utf-8') for x in getattr(group, 'genes').read()]
						break


		def get_ondisk_datalist(self):

			with tables.open_file(self.diskfile, 'r') as f:
				datalist = []
				for group in f.walk_groups():
					try:
						shape = getattr(group, 'shape').read()
						datalist.append([group._v_name,shape])
					except tables.NoSuchNodeError:
						pass
			return datalist

		def get_batch_from_disk(self,group_name,li,hi,barcodes=False):
			with tables.open_file(self.diskfile, 'r') as f:
				for group in f.walk_groups():
					if group_name in group._v_name:
						data = getattr(group, 'data').read()
						indices = getattr(group, 'indices').read()
						indptr = getattr(group, 'indptr').read()
						shape = getattr(group, 'shape').read()

						dat = []
						if len(indptr) < hi: hi = len(indptr)-1
						
						for ci in range(li,hi,1):
							dat.append(np.asarray(csc_matrix((data[indptr[ci]:indptr[ci+1]], indices[indptr[ci]:indptr[ci+1]], np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), shape=(shape[0],1)).todense()).flatten())
						
						if barcodes:
							bc =  [x.decode('utf-8') for x in getattr(group, 'barcodes').read()][li:hi]
							return dat, bc
						else: 
							return dat
	

			
