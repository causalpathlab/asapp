import pandas as pd
import numpy as np
import logging
from typing import Literal
from asap.data.dataloader import DataSet
from asap.model import dcpmf
from asap.model import rpstruct as rp
import asapc
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)

class ASAPNMF:

	def __init__(
		self,
		adata : DataSet,
		tree_max_depth : int = 10,
		num_factors : int = 10
	):
		self.adata = adata
		self.tree_max_depth = tree_max_depth
		self.num_factors = num_factors
	
	def generate_random_projection_mat(self,X_rows):
		rp_mat = []
		for _ in range(self.tree_max_depth):
			rp_mat.append(np.random.normal(size = (X_rows,1)).flatten())                      
		rp_mat = np.asarray(rp_mat)
		logger.info('Random projection matrix :' + str(rp_mat.shape))
		return rp_mat
	
	def generate_pbulk_mat(self,X,rp_mat,batch_label):
		
		logger.info('Running randomizedQR factorization to generate pseudo-bulk data')
		return rp.get_rpqr_psuedobulk(X,rp_mat,batch_label)
		
	def run_nmf(self,batch_iteration):

		rp_mat = self.generate_random_projection_mat(self.adata.shape[0])
		
		if self.adata.run_full_data:
			self._run_nmf_full(rp_mat)
		else:
			self._run_nmf_batch(rp_mat,batch_iteration)

	def _run_nmf_full(self,rp_mat):

		## generate pseudo-bulk
		self.ysum , self.zsum , self.n_bs, self.delta, self.size = self.generate_pbulk_mat(self.adata.mtx, rp_mat,self.adata.batch_label)

		## batch correction model from pseudo-bulk
		pb_model = asapc.ASAPpb(self.ysum,self.zsum,self.delta, self.n_bs,self.n_bs/self.n_bs.sum(0),self.size) 
		pb_res = pb_model.generate_pb()

		## nmf 
		pbulk = np.log1p(pb_res.pb)
		nmf_model = asapc.ASAPdcNMF(pbulk,self.num_factors)
		nmf = nmf_model.nmf()

		## correct batch from dictionary
		u_batch, _, _ = np.linalg.svd(pb_res.batch_effect,full_matrices=False)
		nmf_beta_log = nmf.beta_log - u_batch@u_batch.T@nmf.beta_log

		## predict
		scaler = StandardScaler()
		scaled = scaler.fit_transform(nmf_beta_log)
		reg_model = asapc.ASAPaltNMFPredict(self.adata.mtx,scaled)
		reg = reg_model.predict()

		np.savez(self.adata.outpath+'_dcnmf',
				beta = nmf.beta,
				beta_log = nmf.beta_log,
				theta = reg.theta,
				corr = reg.corr)

	def _run_nmf_batch(self,rp_mat,batch_iteration):

		total_cells = self.adata.shape[1]
		batch_size = self.adata.batch_size

		## just take first batch 
		for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): break

		iend = min(istart + batch_size, total_cells)

		self.adata.load_datainfo_batch(i,istart,iend)
		self.adata.load_data_batch(i,istart,iend)				

		## generate pseudo-bulk
		self.ysum , self.zsum , self.n_bs, self.delta, self.size = self.generate_pbulk_mat(self.adata.mtx, rp_mat,self.adata.batch_label)

		## batch correction model from pseudo-bulk
		pb_model = asapc.ASAPpb(self.ysum,self.zsum,self.delta, self.n_bs,self.n_bs/self.n_bs.sum(0),self.size) 
		pb_res = pb_model.generate_pb()

		## nmf 
		pbulk = np.log1p(pb_res.pb)
		nmf_model = asapc.ASAPdcNMF(pbulk,self.num_factors)
		nmf = nmf_model.nmf()

		if batch_iteration == 1:

			## correct batch from dictionary
			u_batch, _, _ = np.linalg.svd(pb_res.batch_effect,full_matrices=False)
			nmf_beta_log = nmf.beta_log - u_batch@u_batch.T@nmf.beta_log

			## predict
			scaler = StandardScaler()
			scaled = scaler.fit_transform(nmf_beta_log)
			reg_model = asapc.ASAPaltNMFPredict(self.adata.mtx,scaled)
			reg = reg_model.predict()

			np.savez(self.adata.outpath+'_'+str(iend)+'_dcnmf',
					beta = nmf.beta,
					beta_log = nmf.beta_log,
					theta = reg.theta,
					corr = reg.corr,
					barcodes = self.adata.barcodes,
					batch_label = self.adata.batch_label
					)
		else:

			nmf_beta_a = nmf.beta_a
			nmf_beta_b = nmf.beta_b
			nmf_final_beta = None

			for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

				print('running...'+str(i))
				
				## ignore first iteration b/c it is already computed in the above step
				if i ==1 : continue

				iend = min(istart + batch_size, total_cells)

				## clear previous batch
				self.adata.mtx = None
				self.adata.barcodes = None
				self.adata.batch_label = None

				## get current batch
				self.adata.load_datainfo_batch(i,istart,iend)
				self.adata.load_data_batch(i,istart,iend)	

				print(str(self.adata.mtx.shape))	

				if self.adata.mtx.shape[1]<int(batch_size/2):break

				## generate pseudo-bulk
				current_ysum , current_zsum , current_n_bs, current_delta, current_size = self.generate_pbulk_mat(self.adata.mtx, rp_mat,self.adata.batch_label)

				## batch correction model from pseudo-bulk
				pb_model = asapc.ASAPpb(current_ysum,current_zsum,current_delta, current_n_bs,current_n_bs/current_n_bs.sum(0),current_size) 
				pb_res = pb_model.generate_pb()

				
				print(str(pb_res.pb.shape))
				## nmf 
				pbulk = np.log1p(pb_res.pb)

				## here we need to use old model and keep updating beta with new pseudo-bulk data 
				nmf_model_batch = asapc.ASAPdcNMF(pbulk,self.num_factors)
				nmf_batch = nmf_model_batch.online_nmf(nmf_beta_a, nmf_beta_b)
				
				nmf_beta_a = nmf_batch.beta_a
				nmf_beta_b = nmf_batch.beta_b

				nmf_final_beta = nmf_batch.beta
				

			np.savez(self.adata.outpath+'_'+str(iend)+'_dcnmf_final_beta',
					beta = nmf_final_beta
					)
