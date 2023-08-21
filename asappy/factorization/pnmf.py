import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import threading
import queue

from ..tools import DataSet
from ..projection import rpstruct as rp, projection_data
import asapc


import logging
logger = logging.getLogger(__name__)

class asap(object):

	def __init__(
		self,
		adata : DataSet,
		tree_max_depth : int = 10,
		num_factors : int = 10,
		downsample_pseudobulk: bool = False,
		downsample_size: int = 100,
		maxthreads: int = 16,
		num_batch: int = 1
	):
		self.tree_max_depth = tree_max_depth
		self.num_factors = num_factors
		self.downsample_pseudobulk = downsample_pseudobulk
		self.downsample_size = downsample_size
		self.maxthreads = maxthreads
		self.number_batches = num_batch

		logging.info(
			'\nASAP initialized.. \n'+
			'tree depth.. '+str(self.tree_max_depth)+'\n'+
			'number of factors.. '+str(self.num_factors)+'\n'+
			'downsample pseudo-bulk.. '+str(self.downsample_pseudobulk)
		)
	


	def get_barcodes(self):

		if self.adata.uns['run_full_data']:
			return self.adata.obs['barcodes'].values
		else:
			barcodes = []
			for batch_ids in self.pseudobulk_result:
				bi,si,ei = [x for x in batch_ids.keys()][0].split('_')
				barcodes +=  self.adata.load_datainfo_batch(int(bi),int(si),int(ei))
			return barcodes

		
	def get_psuedobulk_batchratio(self,batch_label):
		pb_batch_count = []
		batches = set(batch_label)

		for _,pb_map in self.pseudobulk['pb_map'].items():
			for _,val in pb_map.items():
				pb_batch_count.append([np.count_nonzero(batch_label[val]==x) for x in batches])
		 
		df_pb_batch_count = pd.DataFrame(pb_batch_count).T
		df_pb_batch_count = df_pb_batch_count.T
		df_pb_batch_count.columns = batches
		self.pseudobulk_batchratio = df_pb_batch_count.div(df_pb_batch_count.sum(axis=1), axis=0)

	def get_model_params(self):
		self.adata.uns['shape'] = list(self.adata.uns['shape'])
		model_params = {}
		model_params['tree_max_depth'] = self.tree_max_depth
		model_params['num_factors'] = self.num_factors
		model_params['downsample_pseudobulk'] = self.downsample_pseudobulk
		model_params['downsample_size'] = self.downsample_size
		model_params['number_batches'] = self.number_batches
		return model_params		

	def asap_nmf_predict_batch(self,batch_i,start_index,end_index,beta,result_queue,lock,sema):

		if batch_i <= self.number_batches:

			logging.info('Prediction for batch '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index))
			
			sema.acquire()
			lock.acquire()
			local_mtx = self.adata.load_data_batch(batch_i,start_index,end_index)	
			lock.release()

			reg_model = asapc.ASAPaltNMFPredict(local_mtx.T,beta)
			reg = reg_model.predict()

			result_queue.put({
				str(batch_i) +'_' +str(start_index)+'_'+str(end_index):
				{'theta':reg.theta, 'corr': reg.corr}}
				)
			sema.release()
		
		else:
			logging.info('NO Prediction for batch '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index)+ ' '+str(batch_i) + ' > ' +str(self.number_batches))


	def run_nmf(self):

		logging.info('NMF running...')

		nmf_model = asapc.ASAPdcNMF(self.pseudobulk['pb_data'],self.num_factors)
		self.nmf = nmf_model.nmf()

		scaler = StandardScaler()
		beta_log_scaled = scaler.fit_transform(self.nmf.beta_log)

		total_cells = self.adata.uns['shape'][0]
		batch_size = self.adata.uns['batch_size']

		if total_cells<batch_size:

			logging.info('NMF prediction full data mode...')

			reg_model = asapc.ASAPaltNMFPredict(self.adata.X.T,beta_log_scaled)
			pred = reg_model.predict()
			self.predict_corr = pred.corr
			self.predict_theta = pred.theta
		else:

			logging.info('NMF prediction batch data mode...')

			threads = []
			result_queue = queue.Queue()
			lock = threading.Lock()
			sema = threading.Semaphore(value=self.maxthreads)

			for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

				iend = min(istart + batch_size, total_cells)
								
				thread = threading.Thread(target=self.asap_nmf_predict_batch, args=(i,istart,iend, beta_log_scaled,result_queue,lock,sema))
				
				threads.append(thread)
				thread.start()

			for t in threads:
				t.join()

			predict_result = []
			while not result_queue.empty():
				predict_result.append(result_queue.get())
			

			self.predict_barcodes = []
			for bi,b in enumerate(predict_result):
				for i, (key, value) in enumerate(b.items()):

					batch_index = int(key.split('_')[0])
					start_index = int(key.split('_')[1])
					end_index = int(key.split('_')[2])

					self.predict_barcodes = self.predict_barcodes + self.adata.load_datainfo_batch(batch_index,start_index,end_index)

					if bi ==0 :
						self.predict_theta = value['theta']
						self.predict_corr = value['corr']
					else:
						self.predict_theta = np.vstack((self.predict_theta,value['theta']))
						self.predict_corr = np.vstack((self.predict_corr,value['corr']))
