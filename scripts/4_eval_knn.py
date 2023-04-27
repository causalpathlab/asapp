import sys
import os
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet

import annoy

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


def get_raw_neighbours(dl,nbrsize):	
	nbr_list=[]
	model_ann = ApproxNN(np.asarray(dl.mtx),dl.rows)
	model_ann.build()

	for idx,cell in enumerate(dl.rows):
		neighbours = model_ann.query(np.asarray(dl.mtx[idx,:]).flatten(),k=nbrsize+1)
		## remove self 
		nbr_list.append([ x for x in neighbours if x!=cell])

	return nbr_list



def get_model_neighbours(model_file,mode):

	dl_mtx = model_file['theta']
	# if mode =='full':
	# 	dl_mtx = model_file['theta']
	# else:
	# 	dl_mtx = model_file['corr']

	nbr_list=[]
	model_ann = ApproxNN(dl_mtx,dl.rows)
	model_ann.build()

	for idx,cell in enumerate(dl.rows):
		neighbours = model_ann.query(np.asarray(dl_mtx[idx,:]).flatten(),k=nbrsize+1)
		## remove self 
		nbr_list.append([ x for x in neighbours if x!=cell])

	return nbr_list

def count_common_elements(x,y,nbrsize):
    return np.intersect1d(x,y).size/nbrsize


# sim_data_path = '/home/sishirsubedi/asapp/data/simdata/simdata_r_1.0_s_100_sd_1'
# nmf_data_path = '/home/sishirsubedi/asapp/result/simdata/_r_1.0_s_100_sd_1/'
# altnmf_data = '/home/sishirsubedi/asapp/result/simdata/_r_1.0_s_100_sd_1/_altnmf.npz'
# dcnmf_data = '/home/sishirsubedi/asapp/result/simdata/_r_1.0_s_100_sd_1/_dcnmf.npz'
# fnmf_data = '/home/sishirsubedi/asapp/result/simdata/_r_1.0_s_100_sd_1/_fnmf.npz'
# rho = 1.0
# size = 1000
# seed = 42


sim_data_path = sys.argv[1]
nmf_data_path = sys.argv[2]
altnmf_data = sys.argv[3]
dcnmf_data = sys.argv[4]
fnmf_data = sys.argv[5]
rho = sys.argv[6]
size = int(sys.argv[7])
seed = int(sys.argv[8])

dl = DataSet(sim_data_path,nmf_data_path,data_mode='sparse',data_ondisk=False)
dl.initialize_data()
dl.load_data()

data_rows = list(pd.read_csv(sim_data_path +'.rows.csv.gz' )['rows']) 
result_file = nmf_data_path+'_eval_knn.csv'


nbrsize = int(0.1*size)
raw_nbrs = get_raw_neighbours(dl,nbrsize)
alt_nbrs = get_model_neighbours(np.load(altnmf_data),'alt')
dc_nbrs = get_model_neighbours(np.load(dcnmf_data),'dc')
full_nbrs = get_model_neighbours(np.load(fnmf_data),'full')


alt_res = np.array([ count_common_elements(x,y,nbrsize) for x,y in zip(raw_nbrs,alt_nbrs)]).sum()/size
dc_res = np.array([ count_common_elements(x,y,nbrsize) for x,y in zip(raw_nbrs,dc_nbrs)]).sum()/size
full_res = np.array([ count_common_elements(x,y,nbrsize) for x,y in zip(raw_nbrs,full_nbrs)]).sum()/size

def eval_knn(mode,score):
	result = ['knn',mode,rho,size,seed,score]
	df_res = pd.DataFrame(result).T
	if os.path.isfile(result_file):
		df_res.to_csv(result_file,index=False, mode='a',header=False)
	else:
		df_res.columns = ['method','mode','rho','size','seed','score']
		df_res.to_csv(result_file,index=False)

eval_knn('alt',alt_res)
eval_knn('dc',dc_res)
eval_knn('full',full_res)