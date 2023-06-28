import pandas as  pd
import numpy as np
from scipy.sparse import csc_matrix
import tables
import h5py as h5

class DataSet:
		def __init__(self,inpath,outpath):
			self.inpath = inpath
			self.outpath = outpath

		def initialize_data(self):
			f = h5.File(self.inpath+'.h5', 'r')
			self.sample = list(f.keys())[0]
			self.genes = [x.decode('utf-8') for x in f[self.sample]['gene_names'][()]]
			self.shape = f[self.sample]['shape'][()]
			self.barcodes = [x.decode('utf-8') for x in f[self.sample]['barcodes'][()]]
			f.close()

		def add_batch_label(self,batch_label):
			self.batch_label = batch_label


		def load_data(self,n=0):
			
			# li is starting index 
			# hi is ending index
			li = 0
			if n==0:
				hi = self.shape[1] # load all data
			else:
				hi = n  # load user defined n data
			
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