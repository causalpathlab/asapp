import pandas as pd
import numpy as np
import logging
from asap.data.dataloader import DataSet
from asap.model import nmf,dcpmf
from asap.model import rpstruct as rp
import asapc
from sklearn.preprocessing import StandardScaler
import threading
import queue

logger = logging.getLogger(__name__)

class ASAPNMF:

	def __init__(
		self,
		adata : DataSet,
		tree_max_depth : int = 10,
		num_factors : int = 10,
		downsample_pbulk: bool = False,
		downsample_size: int = 100,
		method: str = 'asap',
		maxthreads: int = 16,
		num_batch: int = 1
	):
		self.adata = adata
		self.tree_max_depth = tree_max_depth
		self.num_factors = num_factors
		self.downsample_pbulk = downsample_pbulk
		self.downsample_size = downsample_size
		self.method = method
		self.maxthreads = maxthreads
		self.number_batches = num_batch

		logging.info(
			'\nASAP initialized.. \n'+
			'tree depth.. '+str(self.tree_max_depth)+'\n'+
			'number of factors.. '+str(self.num_factors)+'\n'+
			'downsample pseudo-bulk.. '+str(self.downsample_pbulk)
		)
	
	def generate_random_projection_mat(self,ndims):
		rp_mat = []
		for _ in range(self.tree_max_depth):
			rp_mat.append(np.random.normal(size = (ndims,1)).flatten())                      
		rp_mat = np.asarray(rp_mat)
		logger.info('Random projection matrix :' + str(rp_mat.shape))
		return rp_mat
	
	def assign_batch_label(self,batch_label):
		self.adata.batch_label = np.array(batch_label)

	def estimate_batch_effect(self,rp_mat):

		logging.info('ASAPNMF estimating batch effect in current data size...')

		rp_mat = self.generate_random_projection_mat(self.adata.shape[1])
		
		## generate pseudo-bulk
		self.ysum , self.zsum , self.n_bs, self.delta, self.size = rp.get_rpqr_psuedobulk_with_bc(self.adata.mtx.T, rp_mat,self.adata.batch_label,self.downsample_pbulk,self.downsample_size)

		logging.info('Batch correction estimate...')
		## batch correction model from pseudo-bulk
		pb_model = asapc.ASAPpb(self.ysum,self.zsum,self.delta, self.n_bs,self.n_bs/self.n_bs.sum(0),self.size) 
		pb_res = pb_model.generate_pb()

		logging.info('Saving model...')

		np.savez(self.adata.outpath+'_pbulk_batcheffect',
				pbulk = pb_res.pb,
				batch_effect = pb_res.batch_effect)
	
	def generate_pseudobulk_batch(self,batch_i,start_index,end_index,rp_mat,result_queue,lock,sema):

		if batch_i <= self.number_batches:

			logging.info('Pseudo-bulk generation for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index))
			
			sema.acquire()

			lock.acquire()
			local_mtx = self.adata.load_data_batch(batch_i,start_index,end_index)	
			lock.release()

			rp.get_rpqr_psuedobulk(local_mtx.T, 
				rp_mat, 
				self.downsample_pbulk,self.downsample_size,
				str(batch_i) +'_' +str(start_index)+'_'+str(end_index),
				result_queue
				)
			sema.release()			
		else:
			logging.info('Pseudo-bulk NOT generated for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index)+ ' '+str(batch_i) + ' > ' +str(self.number_batches))

	def generate_pseudobulk(self):
		
		logging.info('Pseudo-bulk generation...')
		
		total_cells = self.adata.shape[0]
		total_genes = self.adata.shape[1]
		batch_size = self.adata.batch_size

		logging.info('Data size...cell x gene '+str(total_cells) +'x'+ str(total_genes))
		logging.info('Batch size... '+str(batch_size))
		logging.info('Data batch to process... '+str(self.number_batches))

		rp_mat = self.generate_random_projection_mat(self.adata.shape[1])
		
		if total_cells<batch_size:

			## generate pseudo-bulk
			self.pbulk_result = rp.get_rpqr_psuedobulk(self.adata.mtx.T, rp_mat,self.downsample_pbulk,self.downsample_size,'full')

		else:

			threads = []
			result_queue = queue.Queue()
			lock = threading.Lock()
			sema = threading.Semaphore(value=self.maxthreads)

			for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

				iend = min(istart + batch_size, total_cells)
								
				thread = threading.Thread(target=self.generate_pseudobulk_batch, args=(i,istart,iend, rp_mat,result_queue,lock,sema))
				
				threads.append(thread)
				thread.start()

			for t in threads:
				t.join()

			self.pbulk_result = []
			while not result_queue.empty():
				self.pbulk_result.append(result_queue.get())
			
	def filter_pbulk(self,min_size=5):

		logging.info('Pseudo-bulk sample filtering...')

		if len(self.pbulk_result) == 1 and self.adata.run_full_data:
			
			pbulkd = self.pbulk_result['full']['pb_dict'] 

			sample_counts = np.array([len(pbulkd[x])for x in pbulkd.keys()])
			keep_indices = np.where(sample_counts>min_size)[0].flatten() 

			self.pbulk_ysum = self.pbulk_result['full']['pb_data'][:,keep_indices]
			self.pbulk_indices = {key: value for i, (key, value) in enumerate(pbulkd.items()) if i in keep_indices}

		else:
			self.pbulk_indices = {}
			for indx,result_batch in enumerate(self.pbulk_result):

				pbulkd = result_batch[[k for k in result_batch.keys()][0]]['pb_dict']
				ysum = result_batch[[k for k in result_batch.keys()][0]]['pb_data']

				sample_counts = np.array([len(pbulkd[x])for x in pbulkd.keys()])
				keep_indices = np.where(sample_counts>min_size)[0].flatten() 

				ysum = ysum[:,keep_indices]
				pbulkd = {key: value for i, (key, value) in enumerate(pbulkd.items()) if i in keep_indices}

				if indx == 0:
					self.pbulk_ysum = ysum
				else:
					self.pbulk_ysum = np.hstack((self.pbulk_ysum,ysum))
				
				self.pbulk_indices[[k for k in result_batch.keys()][0]] = pbulkd

		logging.info('Pseudo-bulk size :' +str(self.pbulk_ysum.shape))

	def pbulk_batchinfo(self,batch_label):

		pb_batch_count = {}
		batches = set(batch_label)
		for pbid,pbindices in asap.pbulk_indices.items():
			pb_batch_count[pbid] = [np.count_nonzero(asap.adata.batch_label[pbindices]==x) for x in batches]
		 
		df_pb_batch_count = pd.DataFrame(pb_batch_count).T
		df_pb_batch_count = df_pb_batch_count.T
		df_pb_batch_count.columns = batches
		df_pb_batch_count.div(df_pb_batch_count.sum(axis=1), axis=0)

	def get_model_params(self):
		model_params = {}
		model_params['tree_max_depth'] = self.tree_max_depth
		model_params['num_factors'] = self.num_factors
		model_params['batch_size'] = self.adata.batch_size
		model_params['downsample_pseudobulk'] = self.downsample_pbulk
		model_params['downsample_size'] = self.downsample_size
		return model_params


	def run_nmf(self):
		
		if self.method == 'asap':
			self._run_asap_nmf()
		elif self.method == 'cnmf' and self.adata.run_full_data :
			self._run_cnmf_full()
		
	def _run_cnmf_full(self):

		logging.info('ASAPNMF running classical nmf method in full data mode...')

		nmf_model = nmf.mu(self.pbulk_ysum)

		logging.info('Saving model...')

		np.savez(self.adata.outpath+'_classicalnmf',
				beta = nmf_model.W,
				theta = nmf_model.H,
				loss = nmf_model.loss)

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


	def _run_asap_nmf(self):

		logging.info('NMF running...')

		nmf_model = asapc.ASAPdcNMF(np.log1p(self.pbulk_ysum),self.num_factors)
		nmf = nmf_model.nmf()

		scaler = StandardScaler()
		beta_log_scaled = scaler.fit_transform(nmf.beta_log)

		total_cells = self.adata.shape[0]
		batch_size = self.adata.batch_size

		if total_cells<batch_size:

			logging.info('NMF prediction full data mode...')

			reg_model = asapc.ASAPaltNMFPredict(self.adata.mtx.T,beta_log_scaled)
			reg = reg_model.predict()

			logging.info('Saving model...')

			np.savez(self.adata.outpath+'_dcnmf',
	    			params = self.get_model_params(),
					nmf_beta = nmf.beta,
					predict_theta = reg.theta,
					predict_corr = reg.corr)		
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

			logging.info('Saving model...')
			np.savez(self.adata.outpath+'_dcnmf',
	    	    params = self.get_model_params(),
				nmf_beta = nmf.beta,
				predict_barcodes = self.predict_barcodes,
				predict_theta = self.predict_theta,
				predict_corr = self.predict_corr)