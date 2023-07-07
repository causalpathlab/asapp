import pandas as  pd
import numpy as np
from scipy.sparse import csc_matrix
import h5py as hf
import tables


class DataSet:
	def __init__(self,inpath,outpath):
		self.inpath = inpath
		self.outpath = outpath
		self.batch_size = 1000

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
					samples.append(group._v_name)
		return samples 
	
	def initialize_data(self,sample_list,n_per_sample):

		f = hf.File(self.inpath+'.h5', 'r')

		total_samples = len(sample_list)

		if total_samples == 1 and n_per_sample < self.batch_size:
			self.sample_list = sample_list[0]
			self.genes = [x.decode('utf-8') for x in f[self.sample_list[0]]['gene_names'][()]]
			self.shape = f[self.sample_list[0]]['shape'][()]
			self.barcodes = [x.decode('utf-8') for x in f[self.sample_list[0]]['barcodes'][()]]
			f.close()

			self.load_data(0,n_per_sample)

		elif total_samples > 1 and (n_per_sample*total_samples) < self.batch_size:
			## get first dataset for gene list
			self.genes = [x.decode('utf-8') for x in f[sample_list[0]]['gene_names'][()]]
			self.sample_list = sample_list
			barcodes = []
			for sample in self.sample_list:
				barcodes = barcodes + [x.decode('utf-8')+'_'+sample for x in f[sample]['barcodes'][()]][0:n_per_sample]
			self.barcodes = barcodes
			self.shape = [len(self.genes),len(self.barcodes)]
			f.close()

			self.load_data(0,n_per_sample)

		elif total_samples == 1 and n_per_sample > self.batch_size:
		
			self.sample_list = sample_list[0]
			self.genes = [x.decode('utf-8') for x in f[self.sample_list[0]]['gene_names'][()]]
			self.shape = f[self.sample_list[0]]['shape'][()]
			
			self.barcodes = None
			self.mtx = None
			
			f.close()
		
		elif total_samples > 1 and (n_per_sample*total_samples) > self.batch_size:

			## get first dataset for gene list
			self.genes = [x.decode('utf-8') for x in f[sample_list[0]]['gene_names'][()]]
			self.sample_list = sample_list

			len_barcodes = []
			for sample in sample_list:
				len_barcodes += f[sample]['shape'][()][1]
			self.shape = [len(self.genes),len_barcodes]
			
			self.barcodes = None
			self.mtx = None
			
			f.close()

	def add_batch_label(self,batch_label):
		self.batch_label = batch_label


	def load_data(self,start_index, end_index):
		
		if len(self.sample_list) == 1:
						
			with tables.open_file(self.inpath+'.h5', 'r') as f:
				for group in f.walk_groups():
					if self.sample == group._v_name:
						data = getattr(group, 'data').read()
						indices = getattr(group, 'indices').read()
						indptr = getattr(group, 'indptr').read()
						shape = getattr(group, 'shape').read()

						dat = []
						if len(indptr) < hi: hi = len(indptr)-1
						
						for ci in range(start_index,end_index,1):
							dat.append(np.asarray(
							csc_matrix((data[indptr[ci]:indptr[ci+1]], 
							indices[indptr[ci]:indptr[ci+1]], 
							np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
							shape=(shape[0],1)).todense()).flatten())
						
						self.mtx = np.array(dat).T
		else:
						
			with tables.open_file(self.inpath+'.h5', 'r') as f:

				dat = []
				for sample in self.sample_list: # need to loop sample to match initialize data barcode order

					for group in f.walk_groups():
						if sample == group._v_name:
							data = getattr(group, 'data').read()
							indices = getattr(group, 'indices').read()
							indptr = getattr(group, 'indptr').read()
							shape = getattr(group, 'shape').read()

							if len(indptr) < hi: hi = len(indptr)-1
							
							for ci in range(start_index,end_index,1):
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
import pandas as pd
osdm = dm('/data/sishir/data/osteosarcoma/',
[
'BC10', 'BC11', 'BC16', 'BC17', 'BC2', 'BC20', 'BC21', 'BC22', 'BC3', 'BC5', 'BC6'
])
osdm.merge_genes()
osdm.merge_data('osteosarcoma')
'''

'''
from data.dataloader import DataSet as ds
osds = ds('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/data/osteosarcoma/osteosarcoma','/home/BCCRC.CA/ssubedi/projects/experiments/asapp/data/osteosarcoma/osteosarcoma')
sample_list = osds.get_samplenames()

osds.initialize_data(sample_list,n_per_sample=100)
bc = pd.DataFrame(osds.barcodes)
pd.Series([x.split('_')[1] for x in bc[0].values]).value_counts() 
len(bc[0].unique())

osds.load_data(sample_list,n_per_sample=100)
osds.mtx.shape
'''