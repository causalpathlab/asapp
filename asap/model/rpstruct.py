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



def get_rp(mtx,rp_mat):

    logger.info('Randomized QR factorized pseudo-bulk')    
    Z = np.dot(rp_mat,mtx).T
    Q, _ = qr(Z,mode='economic')
    Q = (np.sign(Q) + 1)/2

    df = pd.DataFrame(Q,dtype=int)
    df['code'] = df.astype(str).agg(''.join, axis=1)
    df = df.reset_index()
    df = df[['index','code']]
    return df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']


def get_rpqr_psuedobulk(mtx,rp_mat,batch_label):

    pbulkd = get_rp(mtx,rp_mat)

    bindexd = {}
    batches = list(set(batch_label))

    if len(batches) ==1 :

        logger.info('Generating pseudo-bulk without batch correction')    

        pbulkd = get_rp(mtx,rp_mat)

        pbulk = {}
        for key, value in pbulkd.items():
            pbulk[key] = mtx[:,value].sum(1)

        return pd.DataFrame.from_dict(pbulk,orient='index')

    else:
    
        logger.info('Generating pseudo-bulk with batch correction')    
    
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