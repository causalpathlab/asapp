import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
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



def get_rp_with_bc(mtx,rp_mat,batch_label):

    Z = np.dot(rp_mat,mtx).T

    # batch correction 
    logger.info('Randomized QR factorized pseudo-bulk with regressing out batch effect')    
    b_mat = []
    for b in list(set(batch_label)):
        b_mat.append([ 1 if x == b else 0 for x in batch_label])
    b_mat = np.array(b_mat).T
    

    ## remove batch effect retained in low dimension
    u_batch, _, _ = np.linalg.svd(b_mat,full_matrices=False)
    Zres = Z - u_batch@u_batch.T@Z

    ## correlation before and after removing batch effect
    # print([[np.corrcoef(x,y)[0,1] for x in Z.T] for y in b_mat.T ])
    # print([[np.corrcoef(x,y)[0,1] for x in Zres.T] for y in b_mat.T ])

    Q, _ ,_ = np.linalg.svd(Zres, full_matrices=False)
    Q = (np.sign(Q) + 1)/2


    df = pd.DataFrame(Q,dtype=int)
    df['code'] = df.astype(str).agg(''.join, axis=1)
    df = df.reset_index()
    df = df[['index','code']]
    return Q,df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']

def get_rp(mtx,rp_mat):

    Z = np.dot(rp_mat,mtx)
    _, _, Q = randomized_svd(Z, n_components= Z.shape[0], random_state=0)
    
    scaler = StandardScaler()
    Q = scaler.fit_transform(Q.T)

    Q = (np.sign(Q) + 1)/2
    df = pd.DataFrame(Q,dtype=int)
    df['code'] = df.astype(str).agg(''.join, axis=1)
    df = df.reset_index()
    df = df[['index','code']]
    return Q,df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']


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


def sample_pseudo_bulk(pbulkd,sample_size):
    pbulkd_sample = {}
    for key, value in pbulkd.items():
        if len(value)>sample_size:
            pbulkd_sample[key] = random.sample(value,sample_size)
        else:
            pbulkd_sample[key] = value
    return pbulkd_sample     


def get_rpqr_psuedobulk(mtx,rp_mat,downsample_pbulk,downsample_size,mode,res=None):

    import anndata as an
    import scanpy as sc

    adata = an.AnnData(mtx.T)
    sc.pp.normalize_total(adata,exclude_highly_expressed=True,target_sum=1e6)
    dfm = pd.DataFrame(adata.X)

    # # Q1 = dfm.quantile(0.25)
    # Q3 = dfm.quantile(0.75)
    # IQR = Q3

    # threshold = 1.5
    # upper_bound = Q3 + threshold * IQR

    # for column in dfm.columns:
    #     outliers_upper = dfm[column] > upper_bound[column]
    #     dfm.loc[outliers_upper, column] = upper_bound[column]

    mtx = dfm.to_numpy().T
    
    Q,pbulkd = get_rp(mtx,rp_mat)

    if downsample_pbulk:
        pbulkd = sample_pseudo_bulk(pbulkd,downsample_size)

    ysum_ds = []
    for _, value in pbulkd.items():
        ysum_ds.append(mtx[:,value].mean(1))

    ysum_ds = np.array(ysum_ds).T
    scaler = StandardScaler()
    ysum_ds = np.exp(scaler.fit_transform(np.log1p(ysum_ds)))


    if mode == 'full':
        return {mode:{'pb_data':ysum_ds, 'pb_dict':pbulkd}}
    else:
         res.put({mode:{'pb_data':ysum_ds, 'pb_dict':pbulkd}})
        
def get_rpqr_psuedobulk_with_bc(mtx,rp_mat,batch_label,downsample_pbulk,downsample_size):

    batches = list(set(batch_label))

    logger.info('Number of batches... '+ str(len(batches)))
    
    Q,pbulkd = get_rp_with_bc(mtx,rp_mat,batch_label)

    logger.info('Pseudo-bulk size... '+ str(len(pbulkd)))

    if downsample_pbulk:
        pbulkd = sample_pseudo_bulk(pbulkd,downsample_size)

    ## ysum_ds
    ysum_ds = []
    size_s = []
    for key, value in pbulkd.items():
        size_s.append(len(value))
        ysum_ds.append(mtx[:,value].sum(1))
    ysum_ds = np.array(ysum_ds)

    n_bs = np.zeros((len(batches),len(pbulkd)))
    zsum_ds = np.zeros((len(pbulkd),mtx.shape[0]))
    delta_num_db = np.zeros((len(batches),mtx.shape[0]))
    size_s = np.zeros((len(pbulkd)))

    if len(batches)>1:
        ## zsum_ds
        knn=5
        bindexd = {}
        for b in batches:
            bindexd[b] = [i for i,x in  enumerate(batch_label) if x == b]

        logger.info('Generating batch tree ')
        model_list = get_models(mtx,bindexd)

        zsum_ds = [] 
        for pbi,pb in enumerate(pbulkd.keys()): 
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
            
