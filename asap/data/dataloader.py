import pandas as  pd
import numpy as np
from scipy.sparse import csc_matrix,csr_matrix
import h5py as hf
import tables
import glob
import os



class DataSet:
	def __init__(self,inpath,outpath):
		self.inpath = inpath
		self.outpath = outpath

	def get_datainfo(self):
		with tables.open_file(self.inpath+'.h5', 'r') as f:
			for group in f.walk_groups():
				if '/' not in  group._v_name:
					shape = getattr(group, 'shape').read()
					print(str(group)+'....'+str(shape))

	def get_dataset_names(self):
		datasets = []
		with tables.open_file(self.inpath+'.h5', 'r') as f:
			for group in f.walk_groups():
				if '/' not in  group._v_name:
					datasets.append(group._v_name)
		return datasets
	
	def _estimate_batch_mode(self,dataset_list,batch_size):

		self.batch_size = batch_size
		self.dataset_list = dataset_list
		self.dataset_batch_size = {}

		f = hf.File(self.inpath+'.h5', 'r')

		if len(self.dataset_list) == 1:
			
			n_cells = f[self.dataset_list[0]]['shape'][()][0]

			if n_cells < self.batch_size:
				self.run_full_data = True
				self.dataset_batch_size[self.dataset_list[0]] = n_cells

			else:
				self.run_full_data = False
				self.dataset_batch_size[self.dataset_list[0]] = batch_size
		else:
			n_cells = 0 
			for ds in dataset_list:
				n_cells += f[ds]['shape'][()][0]

			if n_cells < self.batch_size:
				self.run_full_data = True
				for ds in dataset_list:
					self.dataset_batch_size[ds] = f[ds]['shape'][()][0]

			else:
				self.run_full_data = False
				for ds in dataset_list:
					self.dataset_batch_size[ds] = int(((f[ds]['shape'][()][0])/n_cells ) * self.batch_size)
		f.close()

	def initialize_data(self,dataset_list,batch_size):

		self._estimate_batch_mode(dataset_list, batch_size)

		f = hf.File(self.inpath+'.h5', 'r')

		total_datasets = len(dataset_list)

		if total_datasets == 1 and self.run_full_data:

			self.genes = [x.decode('utf-8') for x in f[self.dataset_list[0]]['genes'][()]]
			self.shape = f[self.dataset_list[0]]['shape'][()]
			self.barcodes = [x.decode('utf-8') for x in f[self.dataset_list[0]]['barcodes'][()]]
			self.batch_label = [ self.dataset_list[0] for x in f[self.dataset_list[0]]['barcodes'][()]]
			f.close()

			self.load_full_data()

		elif total_datasets > 1 and self.run_full_data:
			
			## get first dataset for gene list
			self.genes = [x.decode('utf-8') for x in f[dataset_list[0]]['genes'][()]]
			
			barcodes = []
			batch_label = []
			for ds in self.dataset_list:
				start_index = 0
				end_index = self.dataset_batch_size[ds]
				barcodes = barcodes + [x.decode('utf-8')+'@'+ds for x in f[ds]['barcodes'][()]][start_index:end_index]
				batch_label = batch_label + [ds for x in f[ds]['barcodes'][()]][start_index:end_index]
			
			self.barcodes = barcodes
			self.batch_label = batch_label

			self.shape = [len(self.barcodes),len(self.genes)]
			f.close()

			self.load_full_data()

		elif total_datasets == 1 and not self.run_full_data:
		
			self.genes = [x.decode('utf-8') for x in f[self.dataset_list[0]]['genes'][()]]
			self.shape = f[self.dataset_list[0]]['shape'][()]
			
			self.barcodes = None
			self.mtx = None
			
			f.close()
		
		elif total_datasets > 1 and not self.run_full_data:

			## get first dataset for gene list
			self.genes = [x.decode('utf-8') for x in f[dataset_list[0]]['genes'][()]]

			len_barcodes = 0
			for ds in self.dataset_list:
				len_barcodes += f[ds]['shape'][()][0]
			self.shape = [len_barcodes,len(self.genes)]
			
			self.barcodes = None
			self.mtx = None
			
			f.close()

	def load_datainfo_batch(self,batch_index,start_index, end_index):

		f = hf.File(self.inpath+'.h5', 'r')
		
		if len(self.dataset_list) == 1:
			ds = self.dataset_list[0]

			if self.shape[0] < end_index: end_index = self.shape[0] 

			self.barcodes = [x.decode('utf-8')+'@'+ds for x in f[ds]['barcodes'][()]][start_index:end_index]
			self.batch_label = [ds for  x in f[ds]['barcodes'][()]][start_index:end_index]
			f.close()

		else:

			barcodes = []
			batch_label = []
			for ds in self.dataset_list:

				end_index =  self.dataset_batch_size[ds] * batch_index
				start_index = 	end_index - self.dataset_batch_size[ds]			

				dataset_size = f[ds]['shape'][()][0]
				if dataset_size < end_index: end_index = dataset_size

				barcodes = barcodes + [x.decode('utf-8')+'@'+ds for x in f[ds]['barcodes'][()]][start_index:end_index]
				batch_label = batch_label + [ds for x in f[ds]['barcodes'][()]][start_index:end_index]

			self.barcodes = barcodes
			self.batch_label = batch_label
		
			f.close()


	def load_full_data(self):
		
		if len(self.dataset_list) == 1:
			
			start_index = 0
			end_index = self.dataset_batch_size[self.dataset_list[0]]

			with tables.open_file(self.inpath+'.h5', 'r') as f:
				for group in f.walk_groups():
					if self.dataset_list[0] == group._v_name:
						data = getattr(group, 'data').read()
						indices = getattr(group, 'indices').read()
						indptr = getattr(group, 'indptr').read()
						shape = getattr(group, 'shape').read()

						dat = []
						for ci in range(start_index,end_index,1):
							dat.append(np.asarray(
							csr_matrix((data[indptr[ci]:indptr[ci+1]], 
							indices[indptr[ci]:indptr[ci+1]], 
							np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
							shape=(1,shape[1])).todense()).flatten())
						
						self.mtx = np.asarray(dat)
		else:
						
			with tables.open_file(self.inpath+'.h5', 'r') as f:

				# need to loop sample to match initialize data barcode order
				for ds_i,ds in enumerate(self.dataset_list):

					for group in f.walk_groups():
						if  ds == group._v_name:

							start_index = 0
							end_index = self.dataset_batch_size[ds]

							data = getattr(group, 'data').read()
							indices = getattr(group, 'indices').read()
							indptr = getattr(group, 'indptr').read()
							shape = getattr(group, 'shape').read()
							dataset_selected_gene_indices = getattr(group, 'dataset_selected_gene_indices').read()
							
							ds_dat = []
							for ci in range(start_index,end_index,1):
								ds_dat.append(np.asarray(
								csr_matrix((data[indptr[ci]:indptr[ci+1]], 
								indices[indptr[ci]:indptr[ci+1]], 
								np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
								shape=(1,shape[1])).todense()).flatten())
							
							ds_dat = np.asarray(ds_dat)
							ds_dat = ds_dat[:,dataset_selected_gene_indices]
							if ds_i == 0:
								self.mtx = ds_dat
							else:
								self.mtx = np.vstack((self.mtx,ds_dat))


	def load_data_batch(self,batch_index, start_index, end_index):
		
		if len(self.dataset_list) == 1:
			
			with tables.open_file(self.inpath+'.h5', 'r') as f:
				for group in f.walk_groups():
					if self.dataset_list[0] == group._v_name:
						data = getattr(group, 'data').read()
						indices = getattr(group, 'indices').read()
						indptr = getattr(group, 'indptr').read()
						shape = getattr(group, 'shape').read()

						dat = []
						for ci in range(start_index,end_index,1):
							dat.append(np.asarray(
							csr_matrix((data[indptr[ci]:indptr[ci+1]], 
							indices[indptr[ci]:indptr[ci+1]], 
							np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
							shape=(1,shape[1])).todense()).flatten())
						
						self.mtx = np.array(dat)
		else:

			with tables.open_file(self.inpath+'.h5', 'r') as f:

				# need to loop sample to match initialize data barcode order
				for ds_i,ds in enumerate(self.dataset_list): 

					for group in f.walk_groups():
						
						if ds == group._v_name:

							## update index according to each sample 	
							end_index =  self.dataset_batch_size[ds] * batch_index
							start_index = 	end_index - self.dataset_batch_size[ds]			

							data = getattr(group, 'data').read()
							indices = getattr(group, 'indices').read()
							indptr = getattr(group, 'indptr').read()
							shape = getattr(group, 'shape').read()
							dataset_selected_gene_indices = getattr(group, 'dataset_selected_gene_indices').read()
							
							ds_dat = []
							for ci in range(start_index,end_index,1):
								ds_dat.append(np.asarray(
								csr_matrix((data[indptr[ci]:indptr[ci+1]], 
								indices[indptr[ci]:indptr[ci+1]], 
								np.array([0,len(indices[indptr[ci]:indptr[ci+1]])])), 
								shape=(1,shape[1])).todense()).flatten())
							
							ds_dat = np.asarray(ds_dat)
							ds_dat = ds_dat[:,dataset_selected_gene_indices]
							if ds_i == 0:
								self.mtx = ds_dat
							else:
								self.mtx = np.vstack((self.mtx,ds_dat))

class DataMerger:
	def __init__(self,inpath):
		self.inpath = inpath
		self.datasets = glob.glob(inpath+'*.h5ad')


	def get_datainfo(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r')
			print('Dataset : '+ os.path.basename(ds).replace('.h5ad','') 
	 		+' , cells : '+ str(len(f['obs']['_index'])) + 
			', genes : ' + str(f['var']['feature_name']['categories'].shape[0]))
			f.close()

	def merge_genes(self,filter_genes = None):
		final_genes = []
		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
			if ds_i ==0:
				final_genes = set([x.decode('utf-8') for x in list(f['var']['feature_name']['categories']) ])
			else:
				current_genes = set([x.decode('utf-8') for x in list(f['var']['feature_name']['categories']) ])
				final_genes = final_genes.intersection(current_genes)
			f.close()

		self.genes = list(final_genes)
		self.dataset_selected_gene_indices = {}

		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
			current_genes = set([x.decode('utf-8') for x in list(f['var']['feature_name']['categories']) ])
			self.dataset_selected_gene_indices[os.path.basename(ds).replace('.h5ad','')] = [index for index, element in enumerate(current_genes) if element in final_genes]      
			f.close()
	
	def merge_data(self,fname):
	
		for ds_i,ds in enumerate(self.datasets):

			dataset = os.path.basename(ds).replace('.h5ad','')
			dataset_f = hf.File(ds, 'r')

			print('processing...'+dataset)
			
			if ds_i ==0:
				f = hf.File(self.inpath+fname+'.h5','w')
			else:
				f = hf.File(self.inpath+fname+'.h5','a')

			grp = f.create_group(dataset)

			grp.create_dataset('barcodes', data = dataset_f['obs']['_index'] ,compression='gzip')
			grp.create_dataset('genes',data=self.genes,compression='gzip')

			grp.create_dataset('indptr',data=dataset_f['X']['indptr'],compression='gzip')
			grp.create_dataset('indices',data=dataset_f['X']['indices'],compression='gzip')
			grp.create_dataset('data',data=dataset_f['X']['data'],compression='gzip')

			
			data_shape = np.array([dataset_f['obs']['_index'].shape[0],
			dataset_f['var']['feature_name']['categories'].shape[0]])

			grp.create_dataset('shape',data=data_shape)
			
			grp.create_dataset('dataset_selected_gene_indices',data=self.dataset_selected_gene_indices[dataset],compression='gzip')

			f.close()


# from asap.data.dataloader import DataMerger as dm                                                               
# tsdm = dm('data/tabula_sapiens/')                                                                               
# tsdm.get_datainfo()                                                                                             
# ##Dataset : immune_264k , cells : 264824, genes : 58604
# ##Dataset : bc_117k , cells : 117346, genes : 33234
# tsdm.merge_genes()                                                                                              
# tsdm.merge_data('tabula_sapiens')   