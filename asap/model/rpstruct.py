import numpy as np
import pandas as pd
from scipy.linalg import qr
import logging
logger = logging.getLogger(__name__)

import annoy
import random

class ApproxNN():
	def __init__(self, data, labels):
		self.dimension = data.shape[1]
		self.data = data.astype('float32')
		self.labels = labels

	def build(self, number_of_trees=50):
		self.index = annoy.AnnoyIndex(self.dimension,'angular')
		for i, vec in enumerate(self.data):
			self.index.add_item(i, vec.tolist())
		self.index.build(number_of_trees)

	def query(self, vector, k):
		indexes = self.index.get_nns_by_vector(vector.tolist(),k)
		return [self.labels[i] for i in indexes]

def get_models(mtx,bindexd):
	model_list = {}
	for batch in bindexd.keys(): 		
		model_ann = ApproxNN(mtx[:,bindexd[batch]].T,bindexd[batch])
		model_ann.build()
		model_list[batch] = model_ann
	return model_list



def get_rp(mtx,rp_mat,batch_label):

    Z = np.dot(rp_mat,mtx).T

    ## no batch correction 
    # logger.info('Randomized QR factorized pseudo-bulk')    
    Q, _ = qr(Z,mode='economic')
    Q = (np.sign(Q) + 1)/2


    ## batch correction 
    # logger.info('Randomized QR factorized pseudo-bulk with batch correction')    
    # b_mat = []
    # for b in list(set(batch_label)):
    #       b_mat.append([ 1 if x == b else 0 for x in batch_label])
    # b_mat = np.array(b_mat).T
    
    # u_batch, _, _ = np.linalg.svd(b_mat,full_matrices=False)
    # Zres = Z - u_batch@u_batch.T@Z
    # Q, _ ,_ = np.linalg.svd(Zres, full_matrices=False)
    # Q = (np.sign(Q) + 1)/2
    
    df = pd.DataFrame(Q,dtype=int)
    df['code'] = df.astype(str).agg(''.join, axis=1)
    df = df.reset_index()
    df = df[['index','code']]
    return Q,df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']


#### 




# def get_rpqr_psuedobulk_knn(mtx,rp_mat,batch_label):
def get_rpqr_psuedobulk(mtx,rp_mat,batch_label):

    batches = list(set(batch_label))
    
    Q,pbulkd = get_rp(mtx,rp_mat,batch_label)

    if len(batches) >1 :

        logger.info('Generating pseudo-bulk without batch correction')    

        pbulk = {}
        for key, value in pbulkd.items():
            pbulk[key] = mtx[:,value].sum(1)

        return pd.DataFrame.from_dict(pbulk,orient='index')
        
    else:
    
        logger.info('Generating pseudo-bulk with batch correction')    
        bindexd = {}

        for b in batches:
            bindexd[b] = [i for i,x in  enumerate(batch_label) if x == b]

        logger.info('Generating batch tree ')

        model_list = get_models(mtx,bindexd)

        pbulk = {}
        
        for pb in pbulkd.keys():
            
            pb_indxs =  pbulkd[pb]

            if len(pb_indxs) < len(batches):
                continue
            
            n_sample = int(len(pb_indxs)/len(batches))

            sample_d = {}
            for b in batches:
                sample_d[b] = random.sample(pb_indxs,n_sample)

            logger.info(pb)    

            updated_pb = []
            for batch in batches:
                for sample in sample_d[batch]:
                    updated_pb.append(model_list[b].query(mtx[:,sample],k=1)[0])

            ## sum option 
            # pbulk[pb] = mtx[:,updated_pb].sum(1)

            ## sample option
            depth = 10000
            gvals = mtx[:,updated_pb].sum(1)
            gprobs = gvals/gvals.sum() 
            pbulk[pb] = np.random.multinomial(depth,gprobs,1)[0]

        return pd.DataFrame.from_dict(pbulk,orient='index')

def pnb_estimation_genewise(mtx,batch_label):

    import rpy2.robjects as ro
    import rpy2.robjects.packages as rp
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    ro.packages.importr('naivebayes')

    nr,nc = mtx.T.shape
    ro.r.assign("M", ro.r.matrix(mtx.T, nrow=nr, ncol=nc))
    ro.r('colnames(M) <- paste0("V", seq_len(ncol(M)))')
    ro.r.assign('laplace',0.5)
    ro.r.assign('N',np.array(batch_label))
    ro.r('pnb <- poisson_naive_bayes(x=M,y=N,laplace=laplace)')

    return pd.DataFrame(dict(ro.r('coef(pnb)').items()))

def get_rpqr_psuedobulk_ci_genewise(mtx,rp_mat,batch_label):

    ### get naive bayes poisson lambda estimation for each batch
    df_pnb = pnb_estimation_genewise(mtx,batch_label)

    ### get pseudo-bulk groups
    pbulkd = get_rp(mtx,rp_mat,batch_label)
    batches = list(set(batch_label))

    if len(batches) > 1 :

        logger.info('Generating pseudo-bulk with batch correction')    

        pbulk = {}
        for key,vlist in pbulkd.items():
            
            if len(vlist) <= len(batches):
                continue
            else:
                updated_pb = []   
                for value in vlist:
                    pb = mtx[:,value]
                    col = []

                    '''
                    llk-
                    p(gene|batch) = x log(lambda)-lambda
                    '''
                    for b in batches:
                        col.append( pb*np.log(df_pnb[b].values)-df_pnb[b].values )
                    col = np.array(col)
                    
                    '''
                    p(batch|gene) = p(gene|batch_i) / sum_b p(gene_b|batch_b)
                    '''
                    pbg = col[batches.index(batch_label[value])]/col.sum(0)  

                    '''
                    x_prime = (x_g/p_g)/ (1/p_g) 
                    '''
                    updated_pb.append((pb/pbg)/(1/pbg))

            '''
            x_bulk = sum_j x_prime_j
            '''
            pbulk[key] = np.array(updated_pb).sum(0)
        
        logger.info('Generating pseudo-bulk with batch correction -- completed')    

        return pd.DataFrame.from_dict(pbulk,orient='index')


    elif len(batches) ==1 :

        logger.info('Generating pseudo-bulk without batch correction')    

        pbulk = {}
        for key, value in pbulkd.items():
            pbulk[key] = mtx[:,value].sum(1)

        return pd.DataFrame.from_dict(pbulk,orient='index')

def gnb_estimation_rp(q,batch_label):

    from sklearn.naive_bayes import GaussianNB

    dfp = pd.DataFrame()
    for b in  list(set(batch_label)) :
        clf = GaussianNB()
        clf.fit(q, [ 1 if x==b else 0 for x in batch_label])
        dfp[b] = pd.DataFrame(clf.predict_proba(q))[1]
    
    return dfp


def pnb_estimation_rp(mtx,batch_label):

    import rpy2.robjects as ro
    import rpy2.robjects.packages as rp
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    ro.packages.importr('naivebayes')

    nr,nc = mtx.shape
    ro.r.assign("M", ro.r.matrix(mtx, nrow=nr, ncol=nc))
    ro.r('colnames(M) <- paste0("V", seq_len(ncol(M)))')
    ro.r.assign('laplace',0.5)
    ro.r.assign('N',np.array(batch_label))
    ro.r('pnb <- poisson_naive_bayes(x=M,y=N,laplace=laplace)')

    return pd.DataFrame(dict(ro.r('coef(pnb)').items()))


def p_llk(x,lmda):
    return (x * np.log(lmda) + lmda).sum()
    
def calc_delta(x,eta):
    return (x/eta).sum()/(1/eta).sum()

def get_rpqr_psuedobulk_ci_cellwise(mtx,rp_mat,batch_label):


    q,pbulkd = get_rp(mtx,rp_mat,batch_label)
    batches = list(set(batch_label))

    ### gaussian estimation
    logger.info('Generating pseudo-bulk with batch correction genewise ipw using gaussian nb estimation')    
    nb = gnb_estimation_rp(q,batch_label)
    eta = []
    for b in batches:
        eta.append((nb[b]/nb.sum(1)).values)
    eta = pd.DataFrame(eta).T
    eta.columns = batches

    #### poisson estimation 
    # logger.info('Generating pseudo-bulk with batch correction genewise ipw using poisson nb estimation')    
    # df_pnb = pnb_estimation_rp(q,batch_label)
    # nb =[]
    # for b in batches:
    #     nb.append(np.apply_along_axis(p_llk, axis=1, arr=q, lmda=df_pnb[b].values))
    # nb = pd.DataFrame(nb).T
    # nb.columns = batches
    # eta =[]
    # for b in batches:
    #     eta.append((nb[b]/nb.sum(1)).values)
    # eta = pd.DataFrame(eta).T
    # eta.columns = batches

    delta = []
    for b in batches:
        delta.append(np.apply_along_axis(calc_delta, axis=1, arr=mtx, eta=eta[b].values))
    delta = pd.DataFrame(delta).T
    delta.columns = batches
    
    batch_i =[]
    for b in batches:
        batch_i.append( [ 1 if x==b else 0 for x in batch_label])
    batch_i = np.array(batch_i).T

    dm = np.dot(delta.values,batch_i.T)

    pbulk = {}
    for key, value in pbulkd.items():
        pbulk[key] =  mtx[:,value].sum(1) / (dm[:,value].sum(1) + 1e-6)

    logger.info('Generating pseudo-bulk with batch correction genewise ipw completed')    

    return pd.DataFrame.from_dict(pbulk,orient='index')
