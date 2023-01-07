from data import _loader
import pandas as pd
import numpy as np
import logging
import _rpstruct as rp
import _pnmf,_dcnpnmf,_dcpnmf,_dcpnmfb,_dcpnmfv2
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

def run_asapp(sca,min_leaf=10,max_depth=10,save=None):
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

    m = _dcpnmf.DCPoissonMF(n_components=max_depth)    
    m.fit(pd.DataFrame.from_dict(bulk,orient='index').to_numpy())

    m.transform(np.asarray(sca.data.mtx))

    if save:
        save_model(sca,m,save)
    return m


def run_dcasapp(sca,min_leaf=10,max_depth=10,n_components=10,save=None):

    model = _dcnpnmf.DCNPoissonMF()
    model.fit_null(np.asarray(sca.data.mtx))
    dc_mat = np.dot(model.ED,model.EF)
    logger.info('dc correction matrix shape :' + str(dc_mat.shape))

    rp_mat = []
    for i in range(max_depth):
        rp_mat.append(np.random.normal(size = (sca.data.mtx.shape[1],1)).flatten())
    rp_mat = np.asarray(rp_mat)
    logger.info('random projection matrix shape :' + str(rp_mat.shape))

    tree = rp.DCStepTree(sca.data.mtx,rp_mat,dc_mat)

    logger.info('building tree...min_leaf..'+ str(min_leaf)+'..max_depth'+str(max_depth))
    tree.build_tree(min_leaf,max_depth)

    logger.info('making bulk...')
    bulkd = tree.make_bulk()

    sum = 0
    for k in bulkd.keys(): sum += len(bulkd[k])
    logger.info('bulk size : '+str(len(bulkd)) + 'x' + str(sum))    

    bulk = {}
    for key, value in bulkd.items(): 
        bulk[key] = np.asarray(sca.data.mtx[value].sum(0))[0]


    m = _dcpnmf.DCPoissonMF(n_components=n_components)    

    logger.info('running dcpnmf bulk model...n_components..'+str(n_components))
    m.fit(pd.DataFrame.from_dict(bulk,orient='index').to_numpy())

    logger.info('running dcpnmf single cell model...')
    m.transform(np.asarray(sca.data.mtx))

    if save:
        logger.info('saving model...')
        save_model(sca,m,save)
    else:
        return m


def run_scNMF(sca,save=None):
    X = np.asarray(sca.data.mtx)
    model = _dcpnmf.DCPoissonMF(n_components=10,max_iter=50)
    model.fit(X)
    if save:
        save_model(sca,model,save,'sc')
    else:
        return model


def save_model(sca,model,fn,mode='bulk'):
    pd.DataFrame(model.ED,columns=['depth'],index=sca.data.rows).to_csv(fn+'_depth.csv.gz',compression='gzip')
    pd.DataFrame(model.EF.T,columns=['freq'],index=sca.data.cols).to_csv(fn+'_freq.csv.gz',compression='gzip')
    pd.DataFrame(model.Ebeta,columns=sca.data.cols).to_csv(fn+'_beta.csv.gz',compression='gzip')
    pd.DataFrame(model.Etheta,index=sca.data.rows).to_csv(fn+'_theta.csv.gz',compression='gzip')
    pd.DataFrame(model.bound).to_csv(fn+'_bulk_trace.csv.gz',compression='gzip')
    if mode == 'bulk':
        pd.DataFrame(model.bound_sc).to_csv(fn+'_sc_trace.csv.gz',compression='gzip')


