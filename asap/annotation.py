import pandas as pd
import numpy as np
import logging
from typing import Literal
from asap.data.dataloader import DataSet
from asap.model import dcpmf
from asap.model import rpstruct as rp

logger = logging.getLogger(__name__)

class ASAPNMF:

	def __init__(
		self,
		adata : DataSet,
		tree_max_depth : int = 10,
		data_chunk : int = 100000
	):
		self.adata = adata
		self.tree_max_depth = tree_max_depth
		self.chunk_size = data_chunk
	
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
		

	def _generate_pbulk_batch(self,n_samples,rp_mat):

		total_batches = int(n_samples/self.chunk_size)+1
		indices = np.arange(n_samples)
		np.random.shuffle(indices)
		X_shuffled = self.adata.mtx[:,indices]
		batch_label_shuffled = self.adata.batch_label[indices]
		self.pbulk_mat = pd.DataFrame()
		for (i, istart) in enumerate(range(0, n_samples,self.chunk_size), 1):
			iend = min(istart + self.chunk_size, n_samples)
			mini_batch = X_shuffled[:,istart: iend]
			mini_batch_bl = batch_label_shuffled[istart: iend]
			batch_pbulk = self.generate_pbulk_mat(mini_batch, rp_mat,mini_batch_bl)
			self.pbulk_mat = pd.concat([self.pbulk_mat, batch_pbulk], axis=0, ignore_index=True)
			logger.info('completed...' + str(i)+ ' of '+str(total_batches))
		self.pbulk_mat= self.pbulk_mat.to_numpy().T
		logger.info('Final pseudo-bulk matrix :' + str(self.pbulk_mat.shape))


	def _generate_pbulk(self):

		n_samples = self.adata.shape[1]
		rp_mat = self.generate_random_projection_mat(self.adata.shape[0])
		
		if n_samples < self.chunk_size:
			logger.info('Total number is sample ' + str(n_samples) +'..modelling entire dataset')
			
			self.pbulk_mat = self.generate_pbulk_mat(self.adata.mtx, rp_mat,self.adata.batch_label).to_numpy().T

		else:
			logger.info('Total number of sample is ' + str(n_samples) +'..modelling '+str(self.chunk_size) +' chunk of dataset')
			self._generate_pbulk_batch(n_samples,rp_mat)

	def get_pbulk(self):
		self._generate_pbulk()