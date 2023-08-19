import pandas as  pd
import numpy as np
from scipy.sparse import csc_matrix
import h5py as hf
import tables
import glob
import os
import logging
logger = logging.getLogger(__name__)


class CreateDatasetFromH5:

	def __init__(self,inpath):
		self.inpath = inpath
		self.outpath = './results/'
		self.datasets = glob.glob(inpath+'*.h5')


	def get_datainfo(self):
		for ds in self.datasets:
			f = hf.File(ds, 'r')
			print('Dataset : '+ os.path.basename(ds).replace('.h5','') 
	 		+' , cells : '+ str(f['matrix']['barcodes'].shape[0]) + 
			', genes : ' + str(f['matrix']['features']['id'].shape[0]))
			f.close()

	def merge_genes(self,filter_genes = None):
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
		self.dataset_selected_gene_indices = {}

		for ds_i, ds in enumerate(self.datasets):
			f = hf.File(ds, 'r')
			current_genes = set([x.decode('utf-8') for x in list(f['matrix']['features']['id']) ])
			self.dataset_selected_gene_indices[os.path.basename(ds).replace('.h5','')] = [index for index, element in enumerate(current_genes) if element in final_genes]      
			f.close()
	
	def merge_data(self,fname):
	
		for ds_i,ds in enumerate(self.datasets):

			dataset = os.path.basename(ds).replace('.h5','')
			dataset_f = hf.File(ds, 'r')

			print('processing...'+dataset)
			
			if ds_i ==0:
				f = hf.File(self.outpath+fname+'.h5asap','w')
			else:
				f = hf.File(self.outpath+fname+'.h5asap','a')

			grp = f.create_group(dataset)

			grp.create_dataset('barcodes', data = dataset_f['matrix']['barcodes'] ,compression='gzip')
			grp.create_dataset('genes',data=self.genes,compression='gzip')
			
			grp.create_dataset('indptr',data=dataset_f['matrix']['indptr'],compression='gzip')
			grp.create_dataset('indices',data=dataset_f['matrix']['indices'],compression='gzip')
			grp.create_dataset('data',data=dataset_f['matrix']['data'],compression='gzip')

			nc = tuple(dataset_f['matrix']['shape'])[0]
			nr = tuple(dataset_f['matrix']['shape'])[1]
			
			data_shape = np.array([nr,nc])

			grp.create_dataset('shape',data=data_shape)
			
			grp.create_dataset('dataset_selected_gene_indices',data=self.dataset_selected_gene_indices[dataset],compression='gzip')

			f.close()

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
				f = hf.File(self.outpath+fname+'.h5asap','w')
			else:
				f = hf.File(self.outpath+fname+'.h5asap','a')

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

			smat = csc_matrix(df.to_numpy())
			
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


class CreateDatasetFromH5AD:
	def __init__(self,inpath):
		self.inpath = inpath
		self.datasets = glob.glob(inpath+'*.h5ad')

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
				f = hf.File(self.outpath+fname+'.h5asap','w')
			else:
				f = hf.File(self.outpath+fname+'.h5asap','a')

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

def save_dict_to_h5(group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            nested_group = group.create_group(key)
            save_dict_to_h5(nested_group, value)  
        else:
            group.create_dataset(key, data=value)


def write_asap(asap_object):
	f = hf.File(asap_object.adata.uns['inpath']+'.h5asap','a')
	
	grp = f.create_group('pseudobulk')
	grp.create_dataset('pseudobulk_data',data=asap_object.pseudobulk['pb_data'],compression='gzip')

	grp = f.create_group('nmf')
	grp.create_dataset('nmf_beta',data=asap_object.nmf.beta,compression='gzip')

	grp = f.create_group('prediction')
	grp.create_dataset('barcodes',data=asap_object.predict_barcodes,compression='gzip')
	grp.create_dataset('correlation',data=asap_object.predict_corr,compression='gzip')
	grp.create_dataset('theta',data=asap_object.predict_theta,compression='gzip')

	asap_object.adata.uns['model_params']= asap_object.get_model_params()
	grp = f.create_group('uns')
	save_dict_to_h5(grp,asap_object.adata.uns)

	f.close()


def read_config(config_file):
	import yaml
	with open(config_file) as f:
		params = yaml.safe_load(f)
	return params

def pickle_obj(f_name, data):
	import pickle
	pikd = open(f_name + '.pickle', 'wb')
	pickle.dump(data, pikd)
	pikd.close()

def unpickle_obj(f_name):
	import pickle
	pikd = open(f_name, 'rb')
	data = pickle.load(pikd)
	pikd.close()
	return data

def compress_pickle_obj(f_name, data):
	import pickle 
	import bz2
	with bz2.BZ2File(f_name + '.pbz2', 'w') as f:
		pickle.dump(data, f)

def decompress_pickle_obj(f_name):
	import pickle 
	import bz2
	data = bz2.BZ2File(f_name, 'rb')
	data = pickle.load(data)
	return data