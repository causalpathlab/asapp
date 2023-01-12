from data import _loader
import pandas as pd
import numpy as np
import logging
import _rpstruct as rp
import _dcnpnmf,_dcpnmf
np.random.seed(42)

logger = logging.getLogger(__name__)

class FASTSCA:
    def __init__(self):
        self.config = None
        self.data = self.data()
    
    class data:
        def __init__(self):
            self.mtx_indptr = None
            self.mtx_indices = None
            self.mtx_data = None
            self.rows = None
            self.cols = None
            self.mtx = None

    def initdata(self):
        _loader.initialize_data(self)

    def simdata(self,N,K,P):
        return _loader.sim_data(self,N,K,P)

    def loaddata(self):
        _loader.load_data(self)

def run_asapp(sca,min_leaf,max_depth,n_components,max_iter,save=None):
    rp_mat = []
    for i in range(max_depth):
        rp_mat.append(np.random.normal(size = (sca.data.mtx.shape[1],1)).flatten())
    rp_mat = np.asarray(rp_mat)
    logger.info(rp_mat.shape)

    tree = rp.StepTree(sca.data.mtx,rp_mat)
    tree.build_tree(min_leaf,max_depth)
    bulkd = tree.make_bulk()

    sum = 0
    for k in bulkd.keys(): sum += len(bulkd[k])
    logger.info('bulk size : '+str(len(bulkd)) + 'x' + str(sum))    


    bulk = {}
    for key, value in bulkd.items(): 
        bulk[key] = np.asarray(sca.data.mtx[value].sum(0))[0]

    m = _dcpnmf.DCPoissonMF(n_components=n_components,max_iter=max_iter)    
    m.fit(pd.DataFrame.from_dict(bulk,orient='index').to_numpy())

    m.transform(np.asarray(sca.data.mtx),max_iter)

    if save:
        save_model(sca,m,save)
    return m


def run_dcasapp(sca,min_leaf,max_depth,n_components,max_iter,mode,n_pass=None,save=None):

    model = _dcnpnmf.DCNPoissonMF()
    model.fit_null(np.asarray(sca.data.mtx))
    dc_mat = np.dot(model.ED,model.EF)
    logger.info('data matrix :' + str(dc_mat.shape))

    rp_mat = []
    for i in range(max_depth):
        rp_mat.append(np.random.normal(size = (sca.data.mtx.shape[1],1)).flatten())
    rp_mat = np.asarray(rp_mat)
    logger.info('random projection matrix :' + str(rp_mat.shape))

    tree = rp.DCStepTree(sca.data.mtx,rp_mat,dc_mat)

    logger.info('building tree...min_leaf..'+ str(min_leaf)+'..max_depth..'+str(max_depth))
    tree.build_tree(min_leaf,max_depth)

    logger.info('making bulk...')
    bulkd = tree.make_bulk()

    sum = 0
    for k in bulkd.keys(): sum += len(bulkd[k])
    logger.info('bulk matrix : '+str(len(bulkd)) + 'x' + str(sum))    

    bulk = {}
    for key, value in bulkd.items(): 
        bulk[key] = np.asarray(sca.data.mtx[value].sum(0))[0]
    X = pd.DataFrame.from_dict(bulk,orient='index').to_numpy()

    if mode =='batch':
        batch_size = int(X.shape[0]/10)
        model = _dcpnmf.DCPoissonMFB(n_components=n_components,max_iter=max_iter,n_pass=n_pass,batch_size=batch_size)    
        logger.info('running dcpnmf bulk model..with..n_components..'+str(n_components))
        model.fit(X)
        logger.info('running bulk model for single cell theta...')
        model.transform(np.asarray(sca.data.mtx),max_iter*10)

    else:
        model = _dcpnmf.DCPoissonMF(n_components=n_components,max_iter=max_iter)    
        logger.info('running dcpnmf bulk model..with..n_components..'+str(n_components))
        model.fit(X)
        logger.info('running bulk model for single cell theta...')
        model.transform(np.asarray(sca.data.mtx),max_iter)

    if save:
        logger.info('saving bulk model...')
        save_model(sca,model,save)
    else:
        return model


def run_scNMF(sca,n_components,max_iter,mode,batch_size=None,n_pass=None,save=None):
    logger.info('running dcpnmf single cell model...')
    X = np.asarray(sca.data.mtx)

    if mode=='batch':
        model = _dcpnmf.DCPoissonMFB(n_components=n_components,max_iter=max_iter,n_pass=n_pass,batch_size=batch_size)  
        model.fit(X)
        model.transform(np.asarray(sca.data.mtx),max_iter*10)
    else:
        model = _dcpnmf.DCPoissonMF(n_components=n_components,max_iter=max_iter)  
        model.fit(X)
        model.transform(np.asarray(sca.data.mtx),max_iter)

    if save:
        logger.info('saving scNMF model...')
        save_model(sca,model,save,'sc')
    else:
        return model


def save_model(sca,model,fn,mode='bulk'):
    pd.DataFrame(model.ED,columns=['depth'],index=sca.data.rows).to_csv(fn+'_model_depth.csv.gz',compression='gzip')
    pd.DataFrame(model.EF.T,columns=['freq'],index=sca.data.cols).to_csv(fn+'_model_freq.csv.gz',compression='gzip')
    pd.DataFrame(model.Ebeta,columns=sca.data.cols).to_csv(fn+'_model_beta.csv.gz',compression='gzip')
    pd.DataFrame(model.Etheta,index=sca.data.rows).to_csv(fn+'_model_theta.csv.gz',compression='gzip')
    pd.DataFrame(model.bound).to_csv(fn+'_model_bulk_trace.csv.gz',compression='gzip')
    if mode == 'bulk':
        pd.DataFrame(model.bound_sc).to_csv(fn+'_model_sc_trace.csv.gz',compression='gzip')


