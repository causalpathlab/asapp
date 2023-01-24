from model import _dcpmf_aux as _dcpmf
from data._dataloader import DataSet
import pandas as pd
import jax.numpy as jnp
import numpy as np
import logging
from typing import Literal
from model import _rpstruct as rp, _rpqr as rpqr
np.random.seed(42)

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
        pbulk_method : Literal['tree','qr']='tree',
        tree_min_leaf : int = 10,
        tree_max_depth : int = 10,
        factorization : Literal['VB','SVB','MVB']='VB',
        n_components : int = 10,
        max_iter : int = 50,
        max_pred_iter : int = 50,
        n_pass : int = 50,
        batch_size : int = 64,
        data_chunk : int = 10000
    ):
        self.adata = adata
        self.generate_pbulk = generate_pbulk
        self.pbulk_method = pbulk_method
        self.tree_min_leaf = tree_min_leaf
        self.tree_max_depth = tree_max_depth
        self.factorization = factorization
        self.n_components = n_components
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
            rp_mat.append(np.random.normal(size = (X_cols,1)).flatten())                      # DC/QR
            # rp_mat.append(np.random.normal(size = (self.tree_max_depth,1)).flatten())
        rp_mat = np.asarray(rp_mat)
        logger.info('Random projection matrix :' + str(rp_mat.shape))
        return rp_mat
    
   
    def generate_pbulk_mat(self,X,rp_mat,dc_mat):
        
        if self.pbulk_method =='qr':
            logger.info('Running random projection tree and generating pseudo-bulk data')
            rproj = rpqr.RPQR(X,rp_mat,dc_mat)
            return rproj.get_psuedobulk()
        
        elif self.pbulk_method =='tree':
            logger.info('Running random projection tree and generating pseudo-bulk data')
            tree = rp.DCStepTree(X,rp_mat,dc_mat)                                               # DC/QR
            # tree = rp.QRStepTree(X,rp_mat,dc_mat)                                               # DC/QR
            tree.build_tree(self.tree_min_leaf,self.tree_max_depth)
            pbulkd = tree.make_bulk()

            #count total number of cells in tree and create pbulk data
            sum = 0
            pbulk = {}
            for key, value in pbulkd.items():
                sum += len(value) 
                pbulk[key] = np.asarray(self.adata.mtx[value].sum(0))[0]
            logger.info('Total number of cells in the tree : ' + str(sum))    
            
            pbulk_mat = pd.DataFrame.from_dict(pbulk,orient='index').to_numpy()
            logger.info('Pseudo-bulk matrix :' + str(pbulk_mat.shape))
            return pbulk_mat

    def _generate_pbulk(self):
        n_samples = self.adata.mtx.shape[0]
        if n_samples < self.chunk_size:
            logger.info('Total number is sample ' + str(n_samples) +'..modelling entire dataset')
            dc_mat = self.generate_degree_correction_mat(self.adata.mtx)
            rp_mat = self.generate_random_projection_mat(self.adata.mtx.shape[1])
            self.pbulk_mat = self.generate_pbulk_mat(self.adata.mtx, rp_mat,dc_mat)
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
            self.model = _dcpmf.DCPoissonMF(n_components=self.n_components,max_iter=self.max_iter)

        elif self.factorization =='SVB':
            logger.info('Factorization mode...SVB')
            self.model = _dcpmf.DCPoissonMFSVB(n_components=self.n_components,max_iter=self.max_iter,n_pass=self.n_pass,batch_size=self.batch_size)                
        
        elif self.factorization =='MVB':
            logger.info('Factorization mode...MVB')
            self.model = _dcpmf.DCPoissonMFMVB(n_components=self.n_components,max_iter=self.max_iter,n_pass=self.n_pass,batch_size=self.batch_size)    
            

    def factorize(self):
        
        logger.info('Model with aux')
        logger.info(self.__dict__)

        if self.generate_pbulk:
            self._generate_pbulk()

        self._model_setup()

        if self.generate_pbulk:
            logger.info('Modelling pseudo-bulk data with n_components..'+str(self.n_components))
            self.model.fit(self.pbulk_mat)
        else:
            logger.info('Modelling all data with n_components..'+str(self.n_components))
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
            self.model.predicted_params['theta_a'] = np.empty(shape=(0,self.n_components))
            self.model.predicted_params['theta_b'] = np.empty(shape=(0,self.n_components))
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