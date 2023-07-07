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
		tree_max_depth : int = 10
	):
		self.adata = adata
		self.tree_max_depth = tree_max_depth
	
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
		

	def _generate_pbulk_batch(self,rp_mat):

		'''
		one sample more than batch size

		many samples n_per_sample * sample size is more than batch size
		'''

		if len(self.adata.sample_list) == 1:

			for sample in sample_list:
				barcodes = barcodes + [x.decode('utf-8')+'_'+sample for x in f[sample]['barcodes'][()]][0:n_per_sample]


		n_cells = self.adata.shape[1]
		chunk_size = self.adata.batch_size

		total_batches = int(n_cells/chunk_size)+1

		self.ysum = pd.DataFrame()
		self.zsum = pd.DataFrame() 
		self.n_bs = pd.DataFrame()
		self.delta = pd.DataFrame() 
		self.size = pd.DataFrame()

		for (i, istart) in enumerate(range(0, n_cells,self.chunk_size), 1):
			iend = min(istart + chunk_size, n_cells)

			
			self.adata.barcodes = [x.decode('utf-8')+'_'+sample for x in f[sample]['barcodes'][()]][istart:iend]
			
			self.adata.load_data(istart,iend)

			mini_batch = X_shuffled[:,istart: iend]
			mini_batch_bl = batch_label_shuffled[istart: iend]
			

			ysum , zsum , n_bs, delta, size  = self.generate_pbulk_mat(mini_batch, rp_mat,mini_batch_bl)

			self.ysum = pd.concat([self.ysum, ysum], axis=0, ignore_index=True)
			self.zsum = pd.concat([self.zsum, zsum], axis=0, ignore_index=True)
			self.n_bs = pd.concat([self.n_bs, n_bs], axis=0, ignore_index=True)
			self.delta = pd.concat([self.delta, delta], axis=0, ignore_index=True)
			self.size = pd.concat([self.size, size], axis=0, ignore_index=True)


			logger.info('completed...' + str(i)+ ' of '+str(total_batches))
		
		self.pbulk_mat= self.pbulk_mat.to_numpy().T
		
		logger.info('Final pseudo-bulk matrix :' + str(self.pbulk_mat.shape))


	def _generate_pbulk(self):

		n_cells = self.adata.shape[1]
		rp_mat = self.generate_random_projection_mat(self.adata.shape[0])
		
		if self.adata.shape[1] < self.adata.batch_size:

			logger.info('Total number of cells ' + str(n_cells) +'..modelling entire dataset')
			
			self.ysum , self.zsum , self.n_bs, self.delta, self.size = self.generate_pbulk_mat(self.adata.mtx, rp_mat,self.adata.batch_label)

		else:
			logger.info('Total number of cells ' + str(n_cells) +'..modelling '+str(self.chunk_size) +' chunk of dataset')
			self._generate_pbulk_batch(rp_mat)

	def get_pbulk(self):
		self._generate_pbulk()