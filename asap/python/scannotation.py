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
		read_from_disk : bool = False,
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
	
   
	def generate_pbulk_mat(self,X,rp_mat,dc_mat=None):
		
		if self.pbulk_method =='qr':
			logger.info('Running randomizedQR factorization to generate pseudo-bulk data')
			return rp.get_rpqr_psuedobulk(X,rp_mat.T)
		
		elif self.pbulk_method =='tree':
			logger.info('Running random projection tree to generate pseudo-bulk data')
			tree = rp.DCStepTree(X,rp_mat,dc_mat)                                               
			tree.build_tree(self.tree_min_leaf,self.tree_max_depth)
			tree.make_bulk()
			return tree.get_rptree_psuedobulk()


	def _generate_pbulk_batch(self,n_samples):

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

	def _generate_pbulk_ondisk(self):
			
		group_list = self.adata.get_ondisk_datalist()
		
		# divide main chunk size by total groups in dataset
		per_group_batch_size = int(self.chunk_size/len(group_list))

		# decide maximum data size - can be user defined value or lowest/highest data size among group members
		# min_data_size = np.min([x[1][1] for x in group_list])
		min_data_size = 5000
		
		total_batches = int(min_data_size/per_group_batch_size)

		logger.info(str(group_list))
		logger.info('Chunk size : '+str(self.chunk_size))
		logger.info('Per group batch size : '+str(per_group_batch_size))
		logger.info('Max per group data size : '+str(min_data_size))
		logger.info('Total batches : '+str(total_batches))

		batch_low_index = 0
		iter = 0
		self.pbulk_mat = pd.DataFrame()
		while batch_low_index<min_data_size:
			mtx = []
			for group in group_list:
				mtx.append(self.adata.get_batch_from_disk(group[0],batch_low_index,batch_low_index+ per_group_batch_size))

			mtx = np.asarray(mtx)
			mtx = np.reshape(mtx,(mtx.shape[0]*mtx.shape[1],mtx.shape[2]))
			mini_batch = np.asmatrix(mtx)

			if self.pbulk_method == 'tree':
				dc_mat = self.generate_degree_correction_mat(mini_batch)
				rp_mat = self.generate_random_projection_mat(mini_batch.shape[1])
				batch_pbulk = self.generate_pbulk_mat(mini_batch, rp_mat,dc_mat)
			
			elif self.pbulk_method == 'qr':
				rp_mat = self.generate_random_projection_mat(mini_batch.shape[1])
				batch_pbulk = self.generate_pbulk_mat(mini_batch, rp_mat)
			
			
			self.pbulk_mat = pd.concat([self.pbulk_mat, batch_pbulk], axis=0, ignore_index=True)

			iter += 1
			batch_low_index = batch_low_index + per_group_batch_size
			logger.info('Current batch size :' + str(mini_batch.shape))
			logger.info('Current pseudo bulk size :' + str(batch_pbulk.shape))
			logger.info('Completed...' + str(iter)+ ' of '+str(total_batches))
		
		logger.info('Final pseudo-bulk matrix :' + str(self.pbulk_mat.shape))
		self.pbulk_mat= self.pbulk_mat.to_numpy()

	def _generate_pbulk(self):

		if self.read_from_disk:

			logger.info('Reading from disk..')
			self._generate_pbulk_ondisk()

		else:
			n_samples = self.adata.mtx.shape[0]
			if n_samples < self.chunk_size:
				logger.info('Total number is sample ' + str(n_samples) +'..modelling entire dataset')
				dc_mat = self.generate_degree_correction_mat(self.adata.mtx)
				rp_mat = self.generate_random_projection_mat(self.adata.mtx.shape[1])
				self.pbulk_mat = self.generate_pbulk_mat(self.adata.mtx, rp_mat,dc_mat).to_numpy()
			else:
				logger.info('Total number of sample is ' + str(n_samples) +'..modelling '+str(self.chunk_size) +' chunk of dataset')
				self._generate_pbulk_batch(n_samples)


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
	
	def get_pbulk(self):
		self._generate_pbulk()


	def factorize(self):
		
		logger.info(self.__dict__)

		if self.generate_pbulk:
			self._generate_pbulk()

		self._model_setup()

		logger.info(self.model.__dict__)

		if self.generate_pbulk:
			logger.info('Modelling pseudo-bulk data with n_components..'+str(self.tree_max_depth))
			self.model.fit(self.pbulk_mat)
			logger.info('Completed model fitting..')
		else:
			logger.info('Modelling all data with n_components..'+str(self.tree_max_depth))
			self.model.fit(np.asarray(self.adata.mtx))
			logger.info('Completed model fitting..')
		

	def _predict_ondisk(self):

		logger.info('Running prediction - reading from disk..')

		group_list = self.adata.get_ondisk_datalist()
		
		# divide main chunk size by total groups in dataset
		per_group_batch_size = int(self.chunk_size/len(group_list))

		# decide maximum data size - can be user defined value or lowest/highest data size among group members
		# min_data_size = np.min([x[1][1] for x in group_list])
		min_data_size = 5000
		
		total_batches = int(min_data_size/per_group_batch_size)

		logger.info(str(group_list))
		logger.info('Chunk size : '+str(self.chunk_size))
		logger.info('Per group batch size : '+str(per_group_batch_size))
		logger.info('Max per group data size : '+str(min_data_size))
		logger.info('Total batches : '+str(total_batches))

		batch_low_index = 0

		iter = 0

		self.model.predicted_params = {}
		self.model.predicted_params['theta_a'] = np.empty(shape=(0,self.tree_max_depth))
		self.model.predicted_params['theta_b'] = np.empty(shape=(0,self.tree_max_depth))
		self.model.predicted_params['depth_a'] = np.empty(shape=(0,1))
		self.model.predicted_params['depth_b'] = np.empty(shape=(0,1))

		all_barcodes = np.empty(shape=(0,1))
		all_data = np.empty(shape=(0,len(self.adata.cols)))

		while batch_low_index<min_data_size:
			mtx = []
			barcodes = []
			for group in group_list:
				m,bc = self.adata.get_batch_from_disk(group[0],batch_low_index,batch_low_index+ per_group_batch_size,barcodes=True)
				mtx.append(m)
				barcodes.append(bc)

			mtx = np.asarray(mtx)
			mtx = np.reshape(mtx,(mtx.shape[0]*mtx.shape[1],mtx.shape[2]))

			barcodes = np.asarray(barcodes)
			barcodes = np.reshape(barcodes,(barcodes.shape[0]*barcodes.shape[1],1))

			batch_predicted_params = self.model.predict_theta(np.asarray(mtx),self.max_pred_iter)
			self.model.predicted_params['theta_a'] = np.concatenate((self.model.predicted_params['theta_a'], batch_predicted_params['theta_a']), axis=0)
			self.model.predicted_params['theta_b'] = np.concatenate((self.model.predicted_params['theta_b'], batch_predicted_params['theta_b']), axis=0)
			self.model.predicted_params['depth_a'] = np.concatenate((self.model.predicted_params['depth_a'], batch_predicted_params['depth_a']), axis=0)
			self.model.predicted_params['depth_b'] = np.concatenate((self.model.predicted_params['depth_b'], batch_predicted_params['depth_b']), axis=0)

			all_barcodes = np.concatenate((all_barcodes, barcodes), axis=0)
			all_data = np.concatenate((all_data, mtx), axis=0)

			iter += 1
			batch_low_index = batch_low_index + per_group_batch_size
			logger.info('Current batch size :' + str(mtx.shape))
			logger.info('Completed...' + str(iter)+ ' of '+str(total_batches))
		
		self.model.predicted_datarows = list(all_barcodes.flatten())
		self.model.predicted_data = all_data



	def predict(self,X):

		if self.read_from_disk:
			self._predict_ondisk()

		else:
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