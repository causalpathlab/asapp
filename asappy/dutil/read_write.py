import pandas as  pd
import numpy as np
from scipy.sparse import csr_matrix
import h5py as hf
import tables
import json
import glob
import os
import logging
logger = logging.getLogger(__name__)


class CreateDatasetFromH5:

	def __init__(self,inpath,sample):
		self.inpath = inpath
		self.outpath = './results/'
		self.datasets = glob.glob(inpath+sample+'.h5')


	def peek_datasets(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r')
			print('Dataset : '+ os.path.basename(ds).replace('.h5','') 
	 		+' , cells : '+ str(f['matrix']['barcodes'].shape[0]) + 
			', genes : ' + str(f['matrix']['features']['id'].shape[0]))
			f.close()

	def merge_genes(self,select_genes = None):
		final_genes = []
		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
			if ds_i ==0:
				final_genes = set([x.decode('utf-8') for x in list(f['matrix']['features']['id']) ])
			else:
				current_genes = set([x.decode('utf-8') for x in list(f['matrix']['features']['id']) ])
				final_genes = final_genes.intersection(current_genes)
			f.close()

		self.genes = list(final_genes)

		if isinstance(select_genes,list):
			self.genes = [ x for x in self.genes if x in select_genes]

		self.dataset_selected_gene_indices = {}

		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
			current_genes = set([x.decode('utf-8') for x in list(f['matrix']['features']['id']) ])
			self.dataset_selected_gene_indices[os.path.basename(ds).replace('.h5','')] = [index for index, element in enumerate(current_genes) if element in self.genes]      
			f.close()
	
	def merge_data(self,fname):
	
		for ds_i,ds in enumerate(self.datasets):

			dataset = os.path.basename(ds).replace('.h5','')
			dataset_f = hf.File(ds, 'r')

			print('processing...'+dataset)
			
			if ds_i ==0:
				f = hf.File(self.outpath+fname+'.h5','w')
			else:
				f = hf.File(self.outpath+fname+'.h5','a')

			grp = f.create_group(dataset)

			grp.create_dataset('barcodes', data = dataset_f['matrix']['barcodes'] ,compression='gzip')
			grp.create_dataset('genes',data=self.genes,compression='gzip')
			
			grp.create_dataset('indptr',data=dataset_f['matrix']['indptr'],compression='gzip')
			grp.create_dataset('indices',data=dataset_f['matrix']['indices'],compression='gzip')
			grp.create_dataset('data',data=dataset_f['matrix']['data'],compression='gzip')

			nr = len(dataset_f['matrix']['barcodes'])
			nc = len(self.genes)
			
			data_shape = np.array([nr,nc])

			grp.create_dataset('shape',data=data_shape)
			
			grp.create_dataset('dataset_selected_gene_indices',data=self.dataset_selected_gene_indices[dataset],compression='gzip')

			f.close()

	def create_asapdata(self,fname,select_genes=None):
			print('Generating common genes...')
			self.merge_genes(select_genes)
			print('Merging datasets...')
			self.merge_data(fname)
			print('completed.')

class CreateDatasetFromH5AD:
	def __init__(self,inpath,sample):
		self.inpath = inpath
		self.outpath = './results/'
		self.datasets = glob.glob(inpath+sample+'.h5ad')

	def check_label(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r')
			if '_index' not in f['obs'].keys():
				print(os.path.basename(ds))
				print(f['obs'].keys())
			f.close()

	def update_label(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r+')
			if '_index' not in f['obs'].keys():
				print(os.path.basename(ds))
				if 'cell_id' in f['obs'].keys():
					f['obs']['_index'] = f['obs']['cell_id']
				elif 'index' in f['obs'].keys():
					f['obs']['_index'] = f['obs']['index']
			f.close()

	def peek_datasets(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r')
			print('Dataset : '+ os.path.basename(ds).replace('.h5ad','') 
	 		+' , cells : '+ str(len(f['obs']['_index'])) + 
			', genes : ' + str(f['var']['feature_name']['categories'].shape[0]))
			f.close()

	def merge_genes(self,select_genes = None):
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
		if isinstance(select_genes,list):
			self.genes = [ x for x in self.genes if x in select_genes]
		self.dataset_selected_gene_indices = {}

		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
			current_genes = set([x.decode('utf-8') for x in list(f['var']['feature_name']['categories']) ])
			self.dataset_selected_gene_indices[os.path.basename(ds).replace('.h5ad','')] = [index for index, element in enumerate(current_genes) if element in self.genes]      
			f.close()
	
	def merge_data(self,fname):
	
		for ds_i,ds in enumerate(self.datasets):

			dataset = os.path.basename(ds).replace('.h5ad','')
			dataset_f = hf.File(ds, 'r')

			print('processing...'+dataset)
			
			if ds_i ==0:
				f = hf.File(self.outpath+fname+'.h5','w')
			else:
				f = hf.File(self.outpath+fname+'.h5','a')

			grp = f.create_group(dataset)

			grp.create_dataset('barcodes', data = dataset_f['obs']['_index'] ,compression='gzip')
			grp.create_dataset('genes',data=self.genes,compression='gzip')

			grp.create_dataset('indptr',data=dataset_f['X']['indptr'],compression='gzip')
			grp.create_dataset('indices',data=dataset_f['X']['indices'],compression='gzip')
			grp.create_dataset('data',data=dataset_f['X']['data'],compression='gzip')

		
			nr = len(dataset_f['obs']['_index']) 
			nc = len(self.genes)
			
			data_shape = np.array([nr,nc])

			grp.create_dataset('shape',data=data_shape)
			
			grp.create_dataset('dataset_selected_gene_indices',data=self.dataset_selected_gene_indices[dataset],compression='gzip')

			f.close()

	def create_asapdata(self,fname,select_genes = None):
			print('Generating common genes...')
			self.merge_genes(select_genes)
			print('Merging datasets...')
			self.merge_data(fname)
			print('completed.')

class CreateDatasetFromMTX:
	def __init__(self,inpath,sample_names):
		self.inpath = inpath
		self.samples = sample_names

	def merge_genes(self,filter_genes = None):
		final_genes = []
		for si,sample in enumerate(self.samples):
			print('processing...'+sample)
			df = pd.read_csv(self.inpath+sample+'/features.tsv.gz',sep='\t',header=None)
			if si == 0:
				final_genes = set([(x).replace('-','_') + '_' + (y).replace('-','_') for x,y in zip(df[0],df[1])])
			else:	
				current_genes = set([(x).replace('-','_') + '_' + (y).replace('-','_') for x,y in zip(df[0],df[1])])
				final_genes = final_genes.intersection(current_genes)
		if filter_genes != None:
			keep_genes = [x for x in list(final_genes) if x in filter_genes]
		self.genes = keep_genes

	
	def merge_data(self,fname):
	
		from scipy.io import mmread

		for si,sample in enumerate(self.samples):

			print('processing...'+sample)
			
			if si ==0:
				f = hf.File(self.outpath+fname+'.h5','w')
			else:
				f = hf.File(self.outpath+fname+'.h5','a')

			mm = mmread(self.inpath+sample+'/matrix.mtx.gz')
			mtx = mm.todense()
			
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

			smat = csr_matrix(df.to_numpy())
			
			grp = f.create_group(sample)

			grp.create_dataset('barcodes',data=df_cols[0].values,compression='gzip')

			genes = [ str(x) for x in df.index.values]
			gene_names = [ str(x) for x in df.index.values]


			batch_label = [ sample for x in range(len(df.columns))]

			grp.create_dataset('batch_label',data=batch_label,compression='gzip')
			grp.create_dataset('genes',data=genes,compression='gzip')
			grp.create_dataset('gene_names',data=gene_names,compression='gzip')
			grp.create_dataset('indptr',data=smat.indptr,compression='gzip')
			grp.create_dataset('indices',data=smat.indices,compression='gzip')
			grp.create_dataset('data',data=smat.data,dtype=np.int32,compression='gzip')

			arr_shape = np.array([len(f[sample]['genes'][()]),len(f[sample]['barcodes'][()])])

			grp.create_dataset('shape',data=arr_shape)
			
			f.close()

			print('completed.')

def is_csr_or_csc(data, indptr, indices):
	num_rows = len(indptr) - 1
	num_cols = max(indices) + 1

	# Check for CSR format
	if len(data) == len(indices) and len(indptr) == num_rows + 1 and max(indices) < num_cols:
		return "CSR"

	# Check for CSC format
	if len(data) == len(indices) and len(indptr) == num_cols + 1 and max(indices) < num_rows:
		return "CSC"

	return "Not CSR or CSC"

def convertMTXtoH5AD(infile,outfile):

	dataset_f = hf.File(infile, 'r')

	print('processing...'+os.path.basename(infile))
	
	f = hf.File(outfile+'.h5ad','w')

	grp = f.create_group('X')
	grp.create_dataset('indptr',data=dataset_f['matrix']['indptr'],compression='gzip')
	grp.create_dataset('indices',data=dataset_f['matrix']['indices'],compression='gzip')
	grp.create_dataset('data',data=dataset_f['matrix']['data'],compression='gzip')

	grp = f.create_group('obs')
	barcodes = [x.decode('utf-8').replace('@','-') for x in list(dataset_f['matrix']['barcodes']) ]
	grp.create_dataset('_index', data = barcodes,compression='gzip')

	
	grp = f.create_group('var')
	g1 = grp.create_group('feature_name')
	g1.create_dataset('categories',data=dataset_f['matrix']['features']['id'],compression='gzip')
	
	f.close()

def save_model(asap_object):
	import anndata as an
	hgvs = asap_object.adata.var.genes[asap_object.adata.uns['pseudobulk']['pb_hvgs']]
	adata = an.AnnData(shape=(len(asap_object.adata.obs.barcodes),len(hgvs)))
	adata.obs_names = [ x for x in asap_object.adata.obs.barcodes]
	adata.var_names = [ x for x in hgvs]
	
	for key,val in asap_object.adata.uns.items():
		adata.uns[key] = val
	
	adata.varm['beta'] = asap_object.adata.uns['pseudobulk']['pb_beta'] 
	adata.obsm['theta'] = asap_object.adata.obsm['theta']
	adata.obsm['corr'] = asap_object.adata.obsm['corr']

	adata.write_h5ad(asap_object.adata.uns['inpath']+'.h5asap')

def write_h5(fname,rows_names,col_names,smat):

	f = hf.File(fname+'.h5','w')

	grp = f.create_group('matrix')

	grp.create_dataset('barcodes', data = rows_names ,compression='gzip')

	grp.create_dataset('indptr',data=smat.indptr,compression='gzip')
	grp.create_dataset('indices',data=smat.indices,compression='gzip')
	grp.create_dataset('data',data=smat.data,compression='gzip')

	data_shape = np.array([len(rows_names),len(col_names)])
	grp.create_dataset('shape',data=data_shape)
	
	f['matrix'].create_group('features')
	f['matrix']['features'].create_dataset('id',data=col_names,compression='gzip')

	f.close()

def read_config(config_file):
	import yaml
	with open(config_file) as f:
		params = yaml.safe_load(f)
	return params

def data_fileformat():
	fpath = './data/*'
	datasets = glob.glob(fpath)
	ftypes = []
	for f in datasets:
		ftypes.append(f.split('.')[len(f.split('.'))-1])
	if len(np.unique(ftypes)) == 1:
		return ftypes[0]
	else:
		raise ValueError('Multiple file formats in the data directory.')

