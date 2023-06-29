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


def get_rpqr_psuedobulk_knn(mtx,rp_mat,batch_label):

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


def counterfactual_nbr(model_list,batches,mtx,cellidx,batchid,nbr):
    
    #TODO implement weight adjusted counterfactual data
    # method to calculate weight from distances
    #https://github.com/causalpathlab/asapR/blob/main/src/mmutil_match.cc#L6

    cf_cell = []
    for batch in batches:
         if batch != batchid:
            cf_idxs,cf_dist = model_list[batch].index.get_nns_by_vector(mtx[:,cellidx],nbr,include_distances=True)
            cf_cell.append(mtx[:,cf_idxs].mean(1))
    return np.array(cf_cell).mean(0)


     
def get_rpqr_psuedobulk(mtx,rp_mat,batch_label):

    batches = list(set(batch_label))
    
    Q,pbulkd = get_rp(mtx,rp_mat,batch_label)


    ## ysum_ds
    ysum_ds = []
    size_s = []
    for key, value in pbulkd.items():
        size_s.append(len(value))
        ysum_ds.append(mtx[:,value].sum(1))
    ysum_ds = np.array(ysum_ds)


    ## zsum_ds
    knn=5
    
    bindexd = {}
    for b in batches:
        bindexd[b] = [i for i,x in  enumerate(batch_label) if x == b]

    logger.info('Generating batch tree ')
    model_list = get_models(mtx,bindexd)


    zsum_ds = [] 
    n_bs = np.zeros((len(batches),len(pbulkd)))
    for pbi,pb in enumerate(pbulkd.keys()): 
        print(pbi) 
        pb_indxs =  pbulkd[pb] 
        n_cells = len(pb_indxs) 
        n_nbr = np.min([knn,n_cells])
        zsum_sample = [] 
        for cellidx in pb_indxs: 
            n_bs[batches.index(batch_label[cellidx]),pbi] += 1
            zsum_sample.append(counterfactual_nbr(model_list,batches,mtx,cellidx,batch_label[cellidx],n_nbr))
        zsum_ds.append(np.array(zsum_sample).mean(0))
    zsum_ds = np.array(zsum_ds)

    delta_num_db = []         
    for b in batches:
        delta_num_db.append(mtx[:,bindexd[b]].sum(1))
    delta_num_db = np.array(delta_num_db)

    return ysum_ds.T, zsum_ds.T, n_bs, delta_num_db.T, size_s