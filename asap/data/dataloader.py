import pandas as  pd
import numpy as np
from scipy.sparse import csc_matrix
import h5py as hf
import tables


class DataSet:
	def __init__(self,inpath,outpath):
		self.inpath = inpath
		self.outpath = outpath

	def load_datainfo(self):
		with tables.open_file(self.inpath+'.h5', 'r') as f:
			for group in f.walk_groups():
				if '/' not in  group._v_name:
					shape = getattr(group, 'shape').read()
					print(str(group)+'....'+str(shape))

	def get_samplenames(self):
		samples = []
		with tables.open_file(self.inpath+'.h5', 'r') as f:
			for group in f.walk_groups():
				if '/' not in  group._v_name:
					samples.append(group)
		return samples 
	
	def initialize_data(self,sample_list):
		f = hf.File(self.inpath+'.h5', 'r')

		if len(sample_list) == 1:
			self.sample = sample_list[0]
			self.genes = [x.decode('utf-8') for x in f[self.sample]['gene_names'][()]]
			self.shape = f[self.sample]['shape'][()]
			self.barcodes = [x.decode('utf-8') for x in f[self.sample]['barcodes'][()]]
			f.close()
			
		else:

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



class DataMerger:
	def __init__(self,inpath,sample_names):
		self.inpath = inpath
		self.samples = sample_names

	def merge_genes(self):
		final_genes = []
		for si,sample in enumerate(self.samples):
			print('processing...'+sample)
			df = pd.read_csv(self.inpath+sample+'/features.tsv.gz',sep='\t',header=None)
			if si == 0:
				final_genes = set([(x).replace('-','_') + '_' + (y).replace('-','_') for x,y in zip(df[0],df[1])])
			else:	
				current_genes = set([(x).replace('-','_') + '_' + (y).replace('-','_') for x,y in zip(df[0],df[1])])
				final_genes = final_genes.intersection(current_genes)
		self.genes = list(final_genes)

	
	def merge_data(self,fname):
	
		from scipy.io import mmread

		for si,sample in enumerate(self.samples):

			print('processing...'+sample)
			
			if si ==0:
				f = hf.File(self.inpath+fname+'.h5','w')
			else:
				f = hf.File(self.inpath+fname+'.h5','a')

			mm = mmread(self.inpath+sample+'/matrix.mtx.gz')
			mtx = mm.todense()
			smat = csc_matrix(mtx)
			
			df_rows = pd.read_csv(self.inpath+sample+'/features.tsv.gz',sep='\t',header=None)
			df_rows['gene'] = [(x).replace('-','_') + '_' + (y).replace('-','_') for x,y in zip(df_rows[0],df_rows[1])]

			df_cols = pd.read_csv(self.inpath+sample+'/barcodes.tsv.gz',sep='\t',header=None)

			df = pd.DataFrame(mtx)
			df.columns = df_cols[0].values
			df.index = df_rows['gene'].values
			df = df.T
			## filter preselected common genes
			print('pre-selection size..'+str(df.shape))
			df = df[self.genes].T
			print('post-selection size..'+str(df.shape))

			grp = f.create_group(sample)

			grp.create_dataset('barcodes',data=df_cols[0].values)

			genes = [ str(x) for x in df_rows['gene'].values]
			gene_names = [ str(x) for x in df_rows['gene'].values]

			grp.create_dataset('genes',data=genes)
			grp.create_dataset('gene_names',data=gene_names)
			grp.create_dataset('indptr',data=smat.indptr)
			grp.create_dataset('indices',data=smat.indices)
			grp.create_dataset('data',data=smat.data,dtype=np.int32)

			arr_shape = np.array([len(f[sample]['genes'][()]),len(f[sample]['barcodes'][()])])

			grp.create_dataset('shape',data=arr_shape)
			f.close()



'''
from data.dataloader import DataMerger as dm
osdm = dm('/data/sishir/data/osteosarcoma/',
[
'BC10', 'BC11', 'BC16', 'BC17', 'BC2', 'BC20', 'BC21', 'BC22', 'BC3', 'BC5', 'BC6'
])
osdm.merge_genes()
osdm.merge_data('osteosarcoma')
'''