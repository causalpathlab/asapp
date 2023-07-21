import pandas as pd
import numpy as np
import logging
from typing import Literal
from asap.data.dataloader import DataSet
from asap.model import nmf,dcpmf
from asap.model import rpstruct as rp
import asapc
from sklearn.preprocessing import StandardScaler
import threading
import queue

logger = logging.getLogger(__name__)

class ASAPNMF:

	def __init__(
		self,
		adata : DataSet,
		tree_max_depth : int = 10,
		num_factors : int = 10,
		downsample_pbulk: bool = False,
		downsample_size: int = 100,
		method: str = 'asap'
	):
		self.adata = adata
		self.tree_max_depth = tree_max_depth
		self.num_factors = num_factors
		self.downsample_pbulk = downsample_pbulk
		self.downsample_size = downsample_size
		self.method = method
	
	def generate_random_projection_mat(self,X_rows):
		rp_mat = []
		for _ in range(self.tree_max_depth):
			rp_mat.append(np.random.normal(size = (X_rows,1)).flatten())                      
		rp_mat = np.asarray(rp_mat)
		logger.info('Random projection matrix :' + str(rp_mat.shape))
		return rp_mat
	
	def estimate_batch_effect(self,rp_mat):

		logging.info('ASAPNMF estimating batch effect in current data size...')

		rp_mat = self.generate_random_projection_mat(self.adata.shape[1])
		
		## generate pseudo-bulk
		self.ysum , self.zsum , self.n_bs, self.delta, self.size = rp.get_rpqr_psuedobulk_with_bc(self.adata.mtx.T, rp_mat,self.adata.batch_label,self.downsample_pbulk,self.downsample_size)

		logging.info('Batch correction estimate...')
		## batch correction model from pseudo-bulk
		pb_model = asapc.ASAPpb(self.ysum,self.zsum,self.delta, self.n_bs,self.n_bs/self.n_bs.sum(0),self.size) 
		pb_res = pb_model.generate_pb()

		logging.info('Saving model...')

		np.savez(self.adata.outpath+'_pbulk_batcheffect',
				pbulk = pb_res.pb,
				batch_effect = pb_res.batch_effect)
	
	def generate_pseudobulk_batch(self,batch_i,start_index,end_index,rp_mat,result_queue,lock):

		logging.info('Processing_'+str(batch_i) +'_' +str(start_index)+'_'+str(end_index))
		
		lock.acquire()
		local_mtx = self.adata.load_data_batch(batch_i,start_index,end_index)	
		lock.release()

		rp.get_rpqr_psuedobulk(local_mtx.T, 
			rp_mat, 
			self.downsample_pbulk,self.downsample_size,
			str(batch_i) +'_' +str(start_index)+'_'+str(end_index),
			result_queue
			)			

	def generate_pseudobulk(self):

		logging.info('Pseudo-bulk generation...')
		
		total_cells = self.adata.shape[0]
		total_genes = self.adata.shape[1]
		batch_size = self.adata.batch_size

		logging.info('Data size...cell x gene '+str(total_cells) +'x'+ str(total_genes))
		logging.info('Batch size... '+str(batch_size))

		rp_mat = self.generate_random_projection_mat(self.adata.shape[1])
		
		if total_cells<batch_size:

			## generate pseudo-bulk
			self.pb_result = rp.get_rpqr_psuedobulk(self.adata.mtx.T, rp_mat,self.adata.batch_label,self.downsample_pbulk,self.downsample_size,'full')

		else:

			threads = []
			result_queue = queue.Queue()
			lock = threading.Lock()

			for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

				iend = min(istart + batch_size, total_cells)
								
				thread = threading.Thread(target=self.generate_pseudobulk_batch, args=(i,istart,iend, rp_mat,result_queue,lock))
				
				threads.append(thread)
				thread.start()

			for t in threads:
				t.join()

			self.pb_result = []
			while not result_queue.empty():
				self.pb_result.append(result_queue.get())
			

	def filter_pbulk(self,min_size=5):

		if len(self.pb_result) == 1:
			sample_counts = np.array([len(self.pbulkd[x])for x in self.pbulkd.keys()])
			keep_indices = np.where(sample_counts>min_size)[0].flatten() 

			self.ysum = self.ysum[:,keep_indices]
			self.pbulkd = {key: value for i, (key, value) in enumerate(self.pbulkd.items()) if i in keep_indices}

		else:
			self.pbulkd = {}
			for indx,result_batch in enumerate(self.pb_result):

				pbulkd = result_batch[[k for k in result_batch.keys()][0]]['pb_dict']
				ysum = result_batch[[k for k in result_batch.keys()][0]]['pb_data']

				sample_counts = np.array([len(pbulkd[x])for x in pbulkd.keys()])
				keep_indices = np.where(sample_counts>min_size)[0].flatten() 

				ysum = ysum[:,keep_indices]
				pbulkd = {key: value for i, (key, value) in enumerate(pbulkd.items()) if i in keep_indices}

				if indx == 0:
					self.ysum = ysum
				else:
					self.ysum = np.hstack((self.ysum,ysum))
				
				self.pbulkd[[k for k in result_batch.keys()][0]] = pbulkd

	def run_nmf(self):
		
		if self.method == 'asap' and self.adata.run_full_data :
			self._run_asap_nmf_full()
		elif self.method == 'asap' and not self.adata.run_full_data :
			self._run_asap_nmf_batch()
		elif self.method == 'cnmf' and self.adata.run_full_data :
			self._run_cnmf_full()
		
	def _run_cnmf_full(self,rp_mat):

		logging.info('ASAPNMF running classical nmf method in full data mode...')

		nmf_model = nmf.mu(self.ysum)

		logging.info('Saving model...')

		np.savez(self.adata.outpath+'_classicalnmf',
				beta = nmf_model.W,
				theta = nmf_model.H,
				loss = nmf_model.loss)
				
	def _run_asap_nmf_full(self,rp_mat):

		logging.info('ASAPNMF running full data mode...')

		nmf_model = asapc.ASAPdcNMF(np.log1p(self.ysum),self.num_factors)
		nmf = nmf_model.nmf()

		logging.info('Prediction...')
		## predict
		scaler = StandardScaler()
		scaled = scaler.fit_transform(nmf.beta_log)
		reg_model = asapc.ASAPaltNMFPredict(self.adata.mtx.T,scaled)
		reg = reg_model.predict()

		logging.info('Saving model...')

		np.savez(self.adata.outpath+'_dcnmf',
				beta = nmf.beta,
				beta_log = nmf.beta_log,
				theta = reg.theta,
				corr = reg.corr)


		# total_cells = asap.adata.shape[0]
		# batch_size = asap.adata.batch_size


		# threads = []
		# result_queue = queue.Queue()

		# for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

		# 	iend = min(istart + batch_size, total_cells)

		# 	logging.info('Processing_'+str(i) +'_' +str(istart)+'_'+str(iend))

		# 	asap.adata.load_datainfo_batch(i,istart,iend)
		# 	asap.adata.load_data_batch(i,istart,iend)				


		# 	thread = threading.Thread(target=asap.generate_pbulk_mat, args=(asap.adata.mtx.T, rp_mat,asap.adata.batch_label))
		# 	threads.append(thread)
		# 	thread.start()

		# for t in threads:
		# 	t.join()

		# result_list = []
		# while not result_queue.empty():
		# 	result_list.append(result_queue.get())
		
		# self.ysum = result_list
			# # ## generate pseudo-bulk
			# ysum = self.generate_pbulk_mat()

			# if i ==1:
			# 	self.ysum = ysum
			# else:
			# 	self.ysum = np.hstack((self.ysum,ysum))

		# logging.info('No batch correction estimate in this step...')
		# logging.info('NMF..')
		## nmf 
		# pbulk = np.log1p(self.ysum)
		# nmf_model = asapc.ASAPdcNMF(self.ysum,self.num_factors)
		# nmf = nmf_model.nmf()

		# ## correct batch from dictionary
		# logging.info('No batch correction for beta dict in this step...')

		# logging.info('Prediction...')
		# ## predict
		# scaler = StandardScaler()
		# scaled = scaler.fit_transform(nmf.beta_log)
		# reg_model = asapc.ASAPaltNMFPredict(self.adata.mtx,scaled)
		# reg = reg_model.predict()

		# logging.info('Saving model...')

		# np.savez(self.adata.outpath+'_pobc_dcnmf',
		# 		beta = nmf.beta,
		# 		beta_log = nmf.beta_log,
		# 		theta = reg.theta,
		# 		corr = reg.corr)


