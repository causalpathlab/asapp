import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
import tables

class DataSet:
		'''
		On memory - format is csr for efficiency since number of rows will be greater then cols
		On disk - h5 format, requires name and number of features in all groups are same
		'''
		def __init__(self,inpath,outpath,data_mode,data_ondisk):
			self.rows = None
			self.cols = None
			self.mtx = None
			self.data_mode = data_mode
			self.ondisk = data_ondisk			
			self.inpath = inpath
			self.outpath = outpath

			if self.ondisk:
				self.diskfile = self.inpath+'.h5'


		def initialize_data(self):

			if self.ondisk:
				self.get_ondisk_features()
			
			if self.data_mode == 'sparse':

				self.npzarrs = self.inpath+'.npz'
				
				self.rows = list(pd.read_csv(self.inpath +'.rows.csv.gz' )['rows']) 
				self.cols = list(pd.read_csv(self.inpath + '.cols.csv.gz')['cols']) 

			if self.data_mode == 'mtx':
				
				## not flip rows/cols as needed
				rf = self.inpath +'.cols.gz'
				cf = self.inpath +'.rows.gz'
				self.rows = list(pd.read_csv(rf,header=None)[0].values)
				self.cols = list(pd.read_csv(cf,header=None)[0].values)
				 

		def load_data(self):

			if self.data_mode == 'sparse':

				npzarrs = np.load(self.npzarrs,allow_pickle=True)
				mtx_indptr = npzarrs['indptr']
				mtx_indices = npzarrs['indices']
				mtx_data = npzarrs['data']

				mtx_dim = len(self.cols)
				row_ids = self.rows

				rows = csr_matrix((mtx_data,mtx_indices,mtx_indptr),shape=(len(row_ids),mtx_dim))
				self.mtx= rows.todense()

			if self.data_mode == 'mtx':

				from scipy.io import mmread 
				
				mm = mmread(self.inpath+'.mtx.gz')
				self.mtx = mm.todense().T


		def sim_data(self,N,K,P):
			from asap.util import sim
			H = sim.generate_H(N, K)
			W = sim.generate_W(P, K)
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
	

			
