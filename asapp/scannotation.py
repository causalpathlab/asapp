from model import _dcpmf
from data._dataloader import DataSet
import pandas as pd
import numpy as np
import logging
from typing import Literal
from model import _rpstruct as rp
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
        tree_min_leaf : int = 10,
        tree_max_depth : int = 10,
        factorization : Literal['VB','SVB','MVB']='VB',
        n_components : int = 10,
        max_iter : int = 50,
        max_pred_iter : int = 10,
        n_pass : int = 25,
        batch_size : int = 64,
        data_chunk : int = 10000
    ):
        self.adata = adata
        self.generate_pbulk = generate_pbulk
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
            rp_mat.append(np.random.normal(size = (X_cols,1)).flatten())
        rp_mat = np.asarray(rp_mat)
        logger.info('Random projection matrix :' + str(rp_mat.shape))
        return rp_mat
    
    def generate_pbulk_mat(self,X,rp_mat,dc_mat):
        logger.info('Running random projection tree and generating pbulk data')
        tree = rp.DCStepTree(X,rp_mat,dc_mat)
        tree.build_tree(self.tree_min_leaf,self.tree_max_depth)
        pbulkd = tree.make_bulk()

        #count total number of cells in tree and create pbulk data
        sum = 0
        pbulk = {}
        for key, value in pbulkd.items():
            sum += len(value) 
            pbulk[key] = np.asarray(self.adata.mtx[value].sum(0))[0]
        logger.info('Total number of cells in the tree : ' + str(sum))    
        
        pbulk_mat = pd.DataFrame.from_dict(pbulk,orient='index')
        logger.info('pbulk matrix :' + str(pbulk_mat.shape))
        return pbulk_mat

    def _generate_pbulk(self):
        n_samples = self.adata.mtx.shape[0]
        if n_samples < self.chunk_size:
            logger.info('total number is sample is :' + str(n_samples) +'..using entire dataset')
            dc_mat = self.generate_degree_correction_mat(self.adata.mtx)
            rp_mat = self.generate_random_projection_mat(self.adata.mtx.shape[1])
            self.pbulk_mat = self.generate_pbulk_mat(self.adata.mtx, rp_mat,dc_mat).to_numpy()
        else:
            logger.info('total number is sample is :' + str(n_samples) +'..using a chunk of dataset')
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
            logger.info('Final pbulk matrix :' + str(self.pbulk_mat.shape))



    def _model_setup(self):
        if self.factorization =='VB':
            logger.info('factorization mode...VB')
            self.model = _dcpmf.DCPoissonMF(n_components=self.n_components,max_iter=self.max_iter)

        elif self.factorization =='SVB':
            logger.info('factorization mode...SVB')
            self.model = _dcpmf.DCPoissonMFSVB(n_components=self.n_components,max_iter=self.max_iter,n_pass=self.n_pass,batch_size=self.batch_size)                
        
        elif self.factorization =='MVB':
            logger.info('factorization mode...MVB')
            self.model = _dcpmf.DCPoissonMFMVB(n_components=self.n_components,max_iter=self.max_iter,n_pass=self.n_pass,batch_size=self.batch_size)    
            
    def _predict(self):
        n_samples = self.adata.mtx.shape[0]
        if n_samples < self.chunk_size:
            logger.info('prediction : total number of sample is :' + str(n_samples) +'..using entire dataset')
            self.predict_theta,self.predict_depth,self.predict_freq = self.model.predict_theta(np.asarray(self.adata.mtx),self.max_pred_iter)
            self.predict_theta = pd.DataFrame(self.predict_theta)
            self.predict_depth = pd.DataFrame(self.predict_depth)
            self.predict_freq = pd.DataFrame(self.predict_freq)

        else:
            logger.info('prediction : total number of sample is :' + str(n_samples) +'..using a chunk of dataset')
            indices = np.arange(n_samples)
            self.predict_theta = pd.DataFrame()
            self.predict_depth = pd.DataFrame()
            self.predict_freq = pd.DataFrame()
            for (i, istart) in enumerate(range(0, n_samples,self.chunk_size), 1):
                iend = min(istart + self.chunk_size, n_samples)
                mini_batch = self.adata.mtx[indices][istart: iend]
                batch_theta,batch_depth,batch_freq = self.model.predict_theta(np.asarray(mini_batch),self.max_pred_iter)
                batch_theta = pd.DataFrame(batch_theta) 
                batch_depth = pd.DataFrame(batch_depth) 
                batch_freq = pd.DataFrame(batch_freq) 
                self.predict_theta = pd.concat([self.predict_theta, batch_theta], axis=0, ignore_index=True)
                self.predict_depth = pd.concat([self.predict_depth, batch_depth], axis=0, ignore_index=True)
                self.predict_freq = pd.concat([self.predict_freq, batch_freq], axis=0, ignore_index=True)
                logger.info('completed...' + str(i))


    def factorize(self):

        if self.generate_pbulk:
            self._generate_pbulk()

        self._model_setup()

        if self.generate_pbulk:
            logger.info('running bulk model..with..n_components..'+str(self.n_components))
            self.model.fit(self.pbulk_mat)
            logger.info('running bulk model for single cell theta...')
        else:
            logger.info('running sc model..with..n_components..'+str(self.n_components))
            self.model.fit(np.asarray(self.adata.mtx))
            logger.info('running sc model for single cell theta...')
        
        self._predict()


    def save_model(self):
        logger.info('saving model...')
        self.predict_depth.index=self.adata.rows
        self.predict_depth.to_csv(self.adata.outpath+'_model_depth.csv.gz',compression='gzip')
        
        self.predict_freq.columns=self.adata.cols
        self.predict_freq.to_csv(self.adata.outpath+'_model_freq.csv.gz',compression='gzip')
        
        pd.DataFrame(self.model.Ebeta,columns=self.adata.cols).to_csv(self.adata.outpath+'_model_beta.csv.gz',compression='gzip')

        self.predict_theta.index=self.adata.rows
        self.predict_theta.to_csv(self.adata.outpath+'_model_theta.csv.gz',compression='gzip')

        pd.DataFrame(self.model.bound).to_csv(self.adata.outpath+'_model_bulk_trace.csv.gz',compression='gzip')


