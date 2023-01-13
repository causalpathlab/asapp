from util import _dataloader
import pandas as pd
import numpy as np
import logging
from typing import Literal
from model import _dcpmf, _rpstruct as rp
from util._dataloader import DataSet
np.random.seed(42)

logger = logging.getLogger(__name__)

class ASAPP:
    '''
    ASAPP: Annotating large-scale Single-cell data matrix by Approximate Projection in Pseudobulk estimation
    
    Parameters
    ----------
    tree_min_leaf : int
        Minimum number of cells in a node in the random projection tree
    tree_max_depth : int
        Maximum number of levels in the random projection tree
    factorization_mode : str 
        Mode of factorization
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
        tree_min_leaf : int = 10,
        tree_max_depth : int = 10,
        factorization_mode : Literal['batch','all']='batch',
        experiment_mode : Literal['bulk','sc']='bulk',
        n_components : int = 10,
        max_iter : int = 10,
        n_pass : int = 10,
        batch_size : int = 32
    ):
        self.adata = adata
        self.tree_min_leaf = tree_min_leaf
        self.tree_max_depth = tree_max_depth

        self.factorization_mode = factorization_mode
        self.experiment_mode = experiment_mode
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_pass = n_pass
        self.batch_size = batch_size

    
    def generate_degree_correction_mat(self):
        logger.info('Running degree correction null model...')
        null_model = _dcpmf.DCNullPoissonMF()
        null_model.fit_null(np.asarray(self.adata.mtx))
        self.dc_mat = np.dot(null_model.ED,null_model.EF)
        logger.info('Degree correction matrix :' + str(self.dc_mat.shape))


    def generate_random_projection_mat(self):
        rp_mat = []
        for _ in range(self.tree_max_depth):
            rp_mat.append(np.random.normal(size = (self.adata.mtx.shape[1],1)).flatten())
        self.rp_mat = np.asarray(rp_mat)
        logger.info('Random projection matrix :' + str(self.rp_mat.shape))
    
    def generate_bulk_mat(self):
        logger.info('Running random projection tree and generating bulk data')
        tree = rp.DCStepTree(self.adata.mtx,self.rp_mat,self.dc_mat)
        tree.build_tree(self.tree_min_leaf,self.tree_max_depth)
        bulkd = tree.make_bulk()

        #count total number of cells in tree and create bulk data
        sum = 0
        bulk = {}
        for key, value in bulkd.items():
            sum += len(value) 
            bulk[key] = np.asarray(self.adata.mtx[value].sum(0))[0]
        logger.info('Total number of cells in the tree : ' + str(sum))    
        
        self.bulk_mat = pd.DataFrame.from_dict(bulk,orient='index').to_numpy()
        logger.info('Bulk matrix :' + str(self.bulk_mat.shape))

    def generate_bulk(self):
        self.generate_degree_correction_mat()
        self.generate_random_projection_mat()
        self.generate_bulk_mat()

    def factorize(self):
        if self.factorization_mode =='batch':
            logger.info('factorization mode...batch')
            self.model = _dcpmf.DCPoissonMFBatch(n_components=self.n_components,max_iter=self.max_iter,n_pass=self.n_pass,batch_size=self.batch_size)    
            if self.experiment_mode=='bulk':
                logger.info('running bulk model..with..n_components..'+str(self.n_components))
                self.model.fit(self.bulk_mat)
                logger.info('running bulk model for single cell theta...')
                self.model.transform(np.asarray(self.adata.mtx),self.max_iter)
            elif self.experiment_mode=='sc':
                logger.info('running sc model..with..n_components..'+str(self.n_components))
                self.model.fit(np.asarray(self.adata.mtx))
                logger.info('running sc model for single cell theta...')
                self.model.transform(np.asarray(self.adata.mtx),self.max_iter)
            
        else:
            logger.info('factorization mode...all')
            self.model = _dcpmf.DCPoissonMF(n_components=self.n_components,max_iter=self.max_iter)
            if self.experiment_mode=='bulk':    
                logger.info('running bulk model..with..n_components..'+str(self.n_components))
                self.model.fit(self.bulk_mat)
                logger.info('running bulk model for single cell theta...')
                self.model.transform(np.asarray(self.adata.mtx),self.max_iter)
            elif self.experiment_mode=='sc':    
                logger.info('running sc model..with..n_components..'+str(self.n_components))
                self.model.fit(self.bulk_mat)
                logger.info('running sc model for single cell theta...')
                self.model.transform(np.asarray(self.adata.mtx),self.max_iter)



    def save_model(self):
        logger.info('saving model...')
        pd.DataFrame(self.model.ED,columns=['depth'],index=self.adata.rows).to_csv(self.adata.outpath+'_model_depth.csv.gz',compression='gzip')
        pd.DataFrame(self.model.EF.T,columns=['freq'],index=self.adata.cols).to_csv(self.adata.outpath+'_model_freq.csv.gz',compression='gzip')
        pd.DataFrame(self.model.Ebeta,columns=self.adata.cols).to_csv(self.adata.outpath+'_model_beta.csv.gz',compression='gzip')
        pd.DataFrame(self.model.Etheta,index=self.adata.rows).to_csv(self.adata.outpath+'_model_theta.csv.gz',compression='gzip')
        pd.DataFrame(self.model.bound).to_csv(self.adata.outpath+'_model_bulk_trace.csv.gz',compression='gzip')
        if self.experiment_mode == 'bulk':
            pd.DataFrame(self.model.bound_sc).to_csv(self.adata.outpath+'_model_sc_trace.csv.gz',compression='gzip')


