import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
import tables
import h5py as h5

class DataSet:
		def __init__(self,sample,inpath,outpath):
			self.sample = sample
			self.inpath = inpath
			self.outpath = outpath


		def initialize_data(self):
			f = h5.File(self.inpath+'.h5', 'r')
			self.genes = [x.decode('utf-8') for x in f[self.sample]['genes'][()]]
			self.shape = f[self.sample]['shape'][()]
			self.barcodes = [x.decode('utf-8') for x in f[self.sample]['barcodes'][()]]
			f.close()

		def add_batch_label(self,batch_label):
			self.batch_label = batch_label


		def load_data(self,n=0):
			
			li = 0
			
			if n==0:
				hi = self.shape[1]
			else:
				hi = n
			
			group = self.sample
			
			with tables.open_file(self.inpath+'.h5', 'r') as f:
				for group in f.walk_groups():
					if self.sample == group._v_name:
						data = getattr(group, 'data').read()
						indices = getattr(group, 'indices').read()
						indptr = getattr(group, 'indptr').read()
						shape = getattr(group, 'shape').read()

						dat = []
						if len(indptr) < hi: hi = len(indptr)-1
						
						for ci in range(li,hi,1):
							dat.append(np.asarray(
							csc_matrix((data[indptr[ci]:indptr[ci+1]], 
							indices[indptr[ci]:indptr[ci+1]], 
							np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
							shape=(shape[0],1)).todense()).flatten())
						
						self.mtx = np.array(dat).T