import pandas as pd

import numpy as np
import logging

from asappy.dataprep.dataloader import DataSet
from asappy.dataprep.read_write import write_asap
import asappy.projection.rpstruct as rp
from sklearn.preprocessing import StandardScaler
import asapc
import threading
import queue

logger = logging.getLogger(__name__)

class asap(object):

	def __init__(
		self,
		sample : str,
		batch_size : int,
		tree_max_depth : int = 10,
		num_factors : int = 10,
		downsample_pseudobulk: bool = False,
		downsample_size: int = 100,
		maxthreads: int = 16,
		num_batch: int = 1
	):
		self.tree_max_depth = tree_max_depth
		self.num_factors = num_factors
		self.downsample_pseudobulk = downsample_pseudobulk
		self.downsample_size = downsample_size
		self.maxthreads = maxthreads
		self.number_batches = num_batch

		logging.info(
			'\nASAP initialized.. \n'+
			'tree depth.. '+str(self.tree_max_depth)+'\n'+
			'number of factors.. '+str(self.num_factors)+'\n'+
			'downsample pseudo-bulk.. '+str(self.downsample_pseudobulk)
		)

		self.adata = DataSet(sample)
		dataset_list = self.adata.get_dataset_names()
		self.adata.initialize_data(dataset_list=dataset_list,batch_size=batch_size)
	
	def generate_random_projection_data(self,ndims):
		return rp.projection_data(self.tree_max_depth,ndims)
	
	
	def generate_pseudobulk_batch(self,batch_i,start_index,end_index,rp_mat,result_queue,lock,sema):

		if batch_i <= self.number_batches:

			logging.info('Pseudo-bulk generation for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index))
			
			sema.acquire()

			lock.acquire()
			local_mtx = self.adata.load_data_batch(batch_i,start_index,end_index)	
			lock.release()

			rp.get_pseudobulk(local_mtx.T, 
				rp_mat, 
				self.downsample_pseudobulk,self.downsample_size,
				str(batch_i) +'_' +str(start_index)+'_'+str(end_index),
				result_queue
				)
			sema.release()			
		else:
			logging.info('Pseudo-bulk NOT generated for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index)+ ' '+str(batch_i) + ' > ' +str(self.number_batches))

	def generate_pseudobulk(self):
		
		logging.info('Pseudo-bulk generation...')
		
		total_cells = self.adata.uns['shape'][0]
		total_genes = self.adata.uns['shape'][1]
		batch_size = self.adata.uns['batch_size']

		logging.info('Data size...cell x gene '+str(total_cells) +'x'+ str(total_genes))
		logging.info('Batch size... '+str(batch_size))
		logging.info('Data batch to process... '+str(self.number_batches))

		rp_mat = self.generate_random_projection_data(self.adata.uns['shape'][1])
		
		if total_cells<batch_size:

			self.pseudobulk_result = rp.get_pseudobulk(self.adata.X.T, rp_mat,self.downsample_pseudobulk,self.downsample_size,'full')

		else:

			threads = []
			result_queue = queue.Queue()
			lock = threading.Lock()
			sema = threading.Semaphore(value=self.maxthreads)

			for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

				iend = min(istart + batch_size, total_cells)
								
				thread = threading.Thread(target=self.generate_pseudobulk_batch, args=(i,istart,iend, rp_mat,result_queue,lock,sema))
				
				threads.append(thread)
				thread.start()

			for t in threads:
				t.join()

			self.pseudobulk_result = []
			while not result_queue.empty():
				self.pseudobulk_result.append(result_queue.get())
			
	def filter_pseudobulk(self,min_size=5):

		self.pseudobulk = {}

		logging.info('Pseudo-bulk sample filtering...')

		if len(self.pseudobulk_result) == 1 and self.adata.uns['run_full_data']:
			
			pseudobulk_map = self.pseudobulk_result['full']['pb_map'] 

			sample_counts = np.array([len(pseudobulk_map[x])for x in pseudobulk_map.keys()])
			keep_indices = np.where(sample_counts>min_size)[0].flatten() 

			self.pseudobulk['pb_data'] = self.pseudobulk_result['full']['pb_data'][:,keep_indices]
			pseudobulk_indices = {key: value for i, (key, value) in enumerate(pseudobulk_map.items()) if i in keep_indices}
			batch_index = str(1)+'_'+str(0)+'_'+str(self.adata.uns['shape'][0])
			self.pseudobulk['pb_map'] = {batch_index:pseudobulk_indices}

		else:
			self.pseudobulk['pb_map'] = {}
			for indx,result_batch in enumerate(self.pseudobulk_result):

				pseudobulk_map = result_batch[[k for k in result_batch.keys()][0]]['pb_map']
				pb = result_batch[[k for k in result_batch.keys()][0]]['pb_data']

				sample_counts = np.array([len(pseudobulk_map[x])for x in pseudobulk_map.keys()])
				keep_indices = np.where(sample_counts>min_size)[0].flatten() 

				pb = pb[:,keep_indices]
				pseudobulk_map = {key: value for i, (key, value) in enumerate(pseudobulk_map.items()) if i in keep_indices}

				if indx == 0:
					self.pseudobulk['pb_data'] = pb
				else:
					self.pseudobulk['pb_data'] = np.hstack((self.pseudobulk['pb_data'],pb))
				
				self.pseudobulk['pb_map'][[k for k in result_batch.keys()][0]] = pseudobulk_map

		logging.info('Pseudo-bulk size :' +str(self.pseudobulk['pb_data'].shape))

	def get_barcodes(self):

		if self.adata.uns['run_full_data']:
			return self.adata.obs['barcodes'].values
		else:
			barcodes = []
			for batch_ids in self.pseudobulk_result:
				bi,si,ei = [x for x in batch_ids.keys()][0].split('_')
				barcodes +=  self.adata.load_datainfo_batch(int(bi),int(si),int(ei))
			return barcodes

		
	def get_psuedobulk_batchratio(self,batch_label):
		pb_batch_count = []
		batches = set(batch_label)

		for _,pb_map in self.pseudobulk['pb_map'].items():
			for _,val in pb_map.items():
				pb_batch_count.append([np.count_nonzero(batch_label[val]==x) for x in batches])
		 
		df_pb_batch_count = pd.DataFrame(pb_batch_count).T
		df_pb_batch_count = df_pb_batch_count.T
		df_pb_batch_count.columns = batches
		self.pseudobulk_batchratio = df_pb_batch_count.div(df_pb_batch_count.sum(axis=1), axis=0)

	def get_model_params(self):
		model_params = {}
		model_params['tree_max_depth'] = self.tree_max_depth
		model_params['num_factors'] = self.num_factors
		model_params['downsample_pseudobulk'] = self.downsample_pseudobulk
		model_params['downsample_size'] = self.downsample_size
		model_params['number_batches'] = self.number_batches
		return model_params		

	def asap_nmf_predict_batch(self,batch_i,start_index,end_index,beta,result_queue,lock,sema):

		if batch_i <= self.number_batches:

			logging.info('Prediction for batch '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index))
			
			sema.acquire()
			lock.acquire()
			local_mtx = self.adata.load_data_batch(batch_i,start_index,end_index)	
			lock.release()

			reg_model = asapc.ASAPaltNMFPredict(local_mtx.T,beta)
			reg = reg_model.predict()

			result_queue.put({
				str(batch_i) +'_' +str(start_index)+'_'+str(end_index):
				{'theta':reg.theta, 'corr': reg.corr}}
				)
			sema.release()
		
		else:
			logging.info('NO Prediction for batch '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index)+ ' '+str(batch_i) + ' > ' +str(self.number_batches))


	def run_nmf(self):

		logging.info('NMF running...')

		nmf_model = asapc.ASAPdcNMF(self.pseudobulk['pb_data'],self.num_factors)
		self.nmf = nmf_model.nmf()

		scaler = StandardScaler()
		beta_log_scaled = scaler.fit_transform(self.nmf.beta_log)

		total_cells = self.adata.uns['shape'][0]
		batch_size = self.adata.uns['batch_size']

		if total_cells<batch_size:

			logging.info('NMF prediction full data mode...')

			reg_model = asapc.ASAPaltNMFPredict(self.adata.X.T,beta_log_scaled)
			self.prediction = reg_model.predict()

		else:

			logging.info('NMF prediction batch data mode...')

			threads = []
			result_queue = queue.Queue()
			lock = threading.Lock()
			sema = threading.Semaphore(value=self.maxthreads)

			for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

				iend = min(istart + batch_size, total_cells)
								
				thread = threading.Thread(target=self.asap_nmf_predict_batch, args=(i,istart,iend, beta_log_scaled,result_queue,lock,sema))
				
				threads.append(thread)
				thread.start()

			for t in threads:
				t.join()

			predict_result = []
			while not result_queue.empty():
				predict_result.append(result_queue.get())
			

			self.predict_barcodes = []
			for bi,b in enumerate(predict_result):
				for i, (key, value) in enumerate(b.items()):

					batch_index = int(key.split('_')[0])
					start_index = int(key.split('_')[1])
					end_index = int(key.split('_')[2])

					self.predict_barcodes = self.predict_barcodes + self.adata.load_datainfo_batch(batch_index,start_index,end_index)

					if bi ==0 :
						self.predict_theta = value['theta']
						self.predict_corr = value['corr']
					else:
						self.predict_theta = np.vstack((self.predict_theta,value['theta']))
						self.predict_corr = np.vstack((self.predict_corr,value['corr']))

	def save_model(self):
		write_asap(self)
