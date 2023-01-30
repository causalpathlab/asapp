import pandas as pd
import numpy as np
import logging
from typing import Literal
from data._dataloader import DataSet
from model import _dcpmf
from model import _rpstruct as rp

logger = logging.getLogger(__name__)

class ASAPP:
	'''
	ASAPP: Annotating large-scale Single-cell data matrix by Approximate Projection in Pseudobulk estimation
	
	Attributes
	----------
	generate_pbulk : bool
		Generate pseudobulk data for factorization
	tree_min_leaf : int
		Minimum number of cells in a node in the random projection tree
	tree_max_depth : int
		Maximum number of levels in the random projection tree
	factorization : str 
		Mode of factorization
		- VB : Full-dataset variational bayes
		- SVB : Stochastic variational bayes
		- MVB : Memoized variational bayes
	n_components : int
		Number of latent components
	max_iter : int
		Number of iterations for optimization
	n_pass : int
		Number of passes for data in batch optimization
	batch_size : int
		Batch size for batch optimization
	
	'''
	def __init__(
		self,
		adata : DataSet,
		generate_pbulk : bool = True,
		read_from_disk : bool = True,
		pbulk_method : Literal['tree','qr']='qr',
		tree_min_leaf : int = 10,
		tree_max_depth : int = 10,
		factorization : Literal['VB','SVB','MVB']='VB',
		max_iter : int = 50,
		max_pred_iter : int = 50,
		n_pass : int = 50,
		batch_size : int = 64,
		data_chunk : int = 10000
	):
		self.adata = adata
		self.generate_pbulk = generate_pbulk
		self.read_from_disk = read_from_disk
		self.pbulk_method = pbulk_method
		self.tree_min_leaf = tree_min_leaf
		self.tree_max_depth = tree_max_depth
		self.factorization = factorization
		self.max_iter = max_iter
		self.max_pred_iter = max_pred_iter
		self.n_pass = n_pass
		self.batch_size = batch_size
		self.chunk_size = data_chunk

	def generate_degree_correction_mat(self,X):
		logger.info('Running degree correction null model...')
		null_model = _dcpmf.DCNullPoissonMF()
		null_model.fit_null(np.asarray(X))
		dc_mat = np.dot(null_model.ED,null_model.EF)
		logger.info('Degree correction matrix :' + str(dc_mat.shape))
		return dc_mat

	def generate_random_projection_mat(self,X_cols):
		rp_mat = []
		for _ in range(self.tree_max_depth):
			rp_mat.append(np.random.normal(size = (X_cols,1)).flatten())                      
		rp_mat = np.asarray(rp_mat)
		logger.info('Random projection matrix :' + str(rp_mat.shape))
		return rp_mat
	
   
	def generate_pbulk_mat(self,X,rp_mat,dc_mat):
		
		if self.pbulk_method =='qr':
			logger.info('Running randomizedQR factorization to generate pseudo-bulk data')
			return rp.get_rpqr_psuedobulk(X,rp_mat.T)
		
		elif self.pbulk_method =='tree':
			logger.info('Running random projection tree to generate pseudo-bulk data')
			tree = rp.DCStepTree(X,rp_mat,dc_mat)                                               
			tree.build_tree(self.tree_min_leaf,self.tree_max_depth)
			tree.make_bulk()
			return tree.get_rptree_psuedobulk()

	def _generate_pbulk(self):

		if self.read_from_disk:

			# logger.info('Reading from disk '+str(self.chunk_size) +' chunk of dataset')
			logger.info('Reading from disk..')

			group_list = self.adata.get_datalist_ondisk()
			
			logger.info(str(group_list))

			batch_size = int(self.chunk_size/len(group_list))
			min_data_size = np.min([x[1][1] for x in group_list])
			
			total_batches = int(min_data_size/self.chunk_size)
			li = 0
			iter = 0
			self.pbulk_mat = pd.DataFrame()
			while li<min_data_size:
				mtx = []
				for group in group_list:
					if group[1][1]> li + batch_size:
						mtx.append(self.adata.get_batch_from_disk(group[0],li,li+batch_size))

				mtx = np.asarray(mtx)
				mtx = np.reshape(mtx,(mtx.shape[0]*mtx.shape[1],mtx.shape[2]))
				print(mtx.shape)
				mini_batch = np.asmatrix(mtx)
				dc_mat = self.generate_degree_correction_mat(mini_batch)
				rp_mat = self.generate_random_projection_mat(mini_batch.shape[1])
				batch_pbulk = self.generate_pbulk_mat(mini_batch, rp_mat,dc_mat)
				self.pbulk_mat = pd.concat([self.pbulk_mat, batch_pbulk], axis=0, ignore_index=True)

				iter += 1
				li = li + batch_size
				logger.info('completed...' + str(iter)+ ' of '+str(total_batches))
			
			self.pbulk_mat= self.pbulk_mat.to_numpy()


		else:
			n_samples = self.adata.mtx.shape[0]
			if n_samples < self.chunk_size:
				logger.info('Total number is sample ' + str(n_samples) +'..modelling entire dataset')
				dc_mat = self.generate_degree_correction_mat(self.adata.mtx)
				rp_mat = self.generate_random_projection_mat(self.adata.mtx.shape[1])
				self.pbulk_mat = self.generate_pbulk_mat(self.adata.mtx, rp_mat,dc_mat).to_numpy()
			else:
				logger.info('Total number of sample is ' + str(n_samples) +'..modelling '+str(self.chunk_size) +' chunk of dataset')
				total_batches = int(n_samples/self.chunk_size)+1
				indices = np.arange(n_samples)
				np.random.shuffle(indices)
				X_shuffled = self.adata.mtx[indices]
				self.pbulk_mat = pd.DataFrame()
				for (i, istart) in enumerate(range(0, n_samples,self.chunk_size), 1):
					iend = min(istart + self.chunk_size, n_samples)
					mini_batch = X_shuffled[istart: iend]
					dc_mat = self.generate_degree_correction_mat(mini_batch)
					rp_mat = self.generate_random_projection_mat(mini_batch.shape[1])
					batch_pbulk = self.generate_pbulk_mat(mini_batch, rp_mat,dc_mat)
					self.pbulk_mat = pd.concat([self.pbulk_mat, batch_pbulk], axis=0, ignore_index=True)
					logger.info('completed...' + str(i)+ ' of '+str(total_batches))
				self.pbulk_mat= self.pbulk_mat.to_numpy()
				logger.info('Final pseudo-bulk matrix :' + str(self.pbulk_mat.shape))



	def _model_setup(self):
		if self.factorization =='VB':
			logger.info('Factorization mode...VB')
			self.model = _dcpmf.DCPoissonMF(n_components=self.tree_max_depth,max_iter=self.max_iter)

		elif self.factorization =='SVB':
			logger.info('Factorization mode...SVB')
			self.model = _dcpmf.DCPoissonMFSVB(n_components=self.tree_max_depth,max_iter=self.max_iter,n_pass=self.n_pass,batch_size=self.batch_size)                
		
		elif self.factorization =='MVB':
			logger.info('Factorization mode...MVB')
			self.model = _dcpmf.DCPoissonMFMVB(n_components=self.tree_max_depth,max_iter=self.max_iter,n_pass=self.n_pass,batch_size=self.batch_size)    
			

	def factorize(self):
		
		logger.info(self.__dict__)

		if self.generate_pbulk:
			self._generate_pbulk()

		self._model_setup()

		if self.generate_pbulk:
			logger.info('Modelling pseudo-bulk data with n_components..'+str(self.tree_max_depth))
			self.model.fit(self.pbulk_mat)
		else:
			logger.info('Modelling all data with n_components..'+str(self.tree_max_depth))
			self.model.fit(np.asarray(self.adata.mtx))
		

	def predict(self,X):
		n_samples = X.shape[0]
		if n_samples < self.chunk_size:
			logger.info('Prediction : total number of sample is :' + str(n_samples) +'..using all dataset')
			self.model.predicted_params = self.model.predict_theta(np.asarray(X),self.max_pred_iter)
			logger.info('Completed...')
		else:
			logger.info('Prediction : total number of sample is :' + str(n_samples) +'..using '+str(self.chunk_size) +' chunk of dataset')
			indices = np.arange(n_samples)
			total_batches = int(n_samples/self.chunk_size)+1
			self.model.predicted_params = {}
			self.model.predicted_params['theta_a'] = np.empty(shape=(0,self.tree_max_depth))
			self.model.predicted_params['theta_b'] = np.empty(shape=(0,self.tree_max_depth))
			self.model.predicted_params['depth_a'] = np.empty(shape=(0,1))
			self.model.predicted_params['depth_b'] = np.empty(shape=(0,1))
			for (i, istart) in enumerate(range(0, n_samples,self.chunk_size), 1):
				iend = min(istart + self.chunk_size, n_samples)
				mini_batch = X[indices][istart: iend]
				batch_predicted_params = self.model.predict_theta(np.asarray(mini_batch),self.max_pred_iter)
				self.model.predicted_params['theta_a'] = np.concatenate((self.model.predicted_params['theta_a'], batch_predicted_params['theta_a']), axis=0)
				self.model.predicted_params['theta_b'] = np.concatenate((self.model.predicted_params['theta_b'], batch_predicted_params['theta_b']), axis=0)
				self.model.predicted_params['depth_a'] = np.concatenate((self.model.predicted_params['depth_a'], batch_predicted_params['depth_a']), axis=0)
				self.model.predicted_params['depth_b'] = np.concatenate((self.model.predicted_params['depth_b'], batch_predicted_params['depth_b']), axis=0)
				logger.info('Completed...' + str(i)+ ' of '+str(total_batches))