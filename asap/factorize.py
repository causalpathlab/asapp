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
	
	def generate_pbulk_mat(self,X,rp_mat,batch_label,pbulk_method):
		
		logger.info('Running randomizedQR factorization to generate pseudo-bulk data')
		return rp.get_rpqr_psuedobulk(X,rp_mat,batch_label,self.downsample_pbulk,self.downsample_size,pbulk_method)

	def estimate_batch_effect(self,rp_mat):

		logging.info('ASAPNMF estimating batch effect in current data size...')

		rp_mat = self.generate_random_projection_mat(self.adata.shape[1])
		
		## generate pseudo-bulk
		self.ysum , self.zsum , self.n_bs, self.delta, self.size = self.generate_pbulk_mat(self.adata.mtx.T, rp_mat,self.adata.batch_label,'batch_effect')

		logging.info('Batch correction estimate...')
		## batch correction model from pseudo-bulk
		pb_model = asapc.ASAPpb(self.ysum,self.zsum,self.delta, self.n_bs,self.n_bs/self.n_bs.sum(0),self.size) 
		pb_res = pb_model.generate_pb()

		logging.info('Saving model...')

		np.savez(self.adata.outpath+'_pbulk_batcheffect',
				pbulk = pb_res.pb,
				batch_effect = pb_res.batch_effect)
		

	def run_nmf(self):

		logging.info('ASAPNMF running...')
		logging.info('Data size... '+str(self.adata.shape))
		logging.info('Batch size... '+str(self.adata.batch_size))

		rp_mat = self.generate_random_projection_mat(self.adata.shape[1])
		
		if self.method == 'asap' and self.adata.run_full_data :
			self._run_asap_nmf_full(rp_mat)
		elif self.method == 'asap' and not self.adata.run_full_data :
			self._run_asap_nmf_batch(rp_mat)
		elif self.method == 'cnmf' and self.adata.run_full_data :
			self._run_cnmf_full(rp_mat)
		
	def _run_cnmf_full(self,rp_mat):

		logging.info('ASAPNMF running classical nmf method in full data mode...')

		## generate pseudo-bulk
		self.ysum = self.generate_pbulk_mat(self.adata.mtx.T, rp_mat,self.adata.batch_label,'nmf')

		logging.info('NMF..')
		## nmf 
		nmf_model = nmf.mu(self.ysum)

		logging.info('Saving model...')

		np.savez(self.adata.outpath+'_classicalnmf',
				beta = nmf_model.W,
				theta = nmf_model.H,
				loss = nmf_model.loss)
				
	def _run_asap_nmf_full(self,rp_mat):

		logging.info('ASAPNMF running full data mode...')

		## generate pseudo-bulk
		self.ysum = self.generate_pbulk_mat(self.adata.mtx.T, rp_mat,self.adata.batch_label,'nmf')

		logging.info('NMF..')
		## nmf 
		pbulk = np.log1p(self.ysum)
		nmf_model = asapc.ASAPdcNMF(self.ysum,self.num_factors)
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

	def _run_asap_nmf_batch(self,rp_mat):

		logging.info('ASAPNMF running pobc - post nmf batch correction batch data mode...')

		total_cells = self.adata.shape[0]
		batch_size = self.adata.batch_size

		threads = []
		result_queue = queue.Queue()

		for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

			logging.info('Processing %s, %s, %s,', (i,istart,iend))


			iend = min(istart + batch_size, total_cells)

			self.adata.load_datainfo_batch(i,istart,iend)
			self.adata.load_data_batch(i,istart,iend)				


			thread = threading.Thread(target=self.generate_pbulk_mat, args=(self.adata.mtx.T, rp_mat,self.adata.batch_label))
			threads.append(thread)
			thread.start()

		for t in threads:
			t.join()

		result_list = []
		while not result_queue.empty():
			result_list.append(result_queue.get())
		
		self.ysum = result_list


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


