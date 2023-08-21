import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import threading
import queue

import asapc


import logging
logger = logging.getLogger(__name__)

def asap_nmf_predict_batch(asap_object,batch_i,start_index,end_index,beta,result_queue,lock,sema):

	if batch_i <= asap_object.number_batches:

		logging.info('Prediction for batch '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index))
		
		sema.acquire()
		lock.acquire()
		local_mtx = asap_object.adata.load_data_batch(batch_i,start_index,end_index)	
		lock.release()

		reg_model = asapc.ASAPaltNMFPredict(local_mtx.T,beta)
		reg = reg_model.predict()

		result_queue.put({
			str(batch_i) +'_' +str(start_index)+'_'+str(end_index):
			{'theta':reg.theta, 'corr': reg.corr}}
			)
		sema.release()
	
	else:
		logging.info('NO Prediction for batch '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index)+ ' '+str(batch_i) + ' > ' +str(asap_object.number_batches))

def asap_nmf(asap_object,num_factors,maxthreads=16):
		
		asap_object.num_factors = num_factors

		logging.info('NMF running...')

		nmf_model = asapc.ASAPdcNMF(asap_object.pseudobulk['pb_data'],asap_object.num_factors)
		nmfres = nmf_model.nmf()

		scaler = StandardScaler()
		beta_log_scaled = scaler.fit_transform(nmfres.beta_log)

		total_cells = asap_object.adata.uns['shape'][0]
		batch_size = asap_object.adata.uns['batch_size']
		
		asap_object.nmf = {}
		asap_object.nmf['beta'] = nmfres.beta

		if total_cells<batch_size:

			logging.info('NMF prediction full data mode...')

			pred_model = asapc.ASAPaltNMFPredict(asap_object.adata.X.T,beta_log_scaled)
			pred = pred_model.predict()
			
			asap_object.nmf['corr'] = pred.corr
			asap_object.nmf['theta'] = pred.theta
		else:

			logging.info('NMF prediction batch data mode...')

			threads = []
			result_queue = queue.Queue()
			lock = threading.Lock()
			sema = threading.Semaphore(value=maxthreads)

			for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

				iend = min(istart + batch_size, total_cells)
								
				thread = threading.Thread(target=asap_nmf_predict_batch, args=(asap_object,i,istart,iend, beta_log_scaled,result_queue,lock,sema))
				
				threads.append(thread)
				thread.start()

			for t in threads:
				t.join()

			predict_result = []
			while not result_queue.empty():
				predict_result.append(result_queue.get())
			

			predict_barcodes = []
			for bi,b in enumerate(predict_result):
				for i, (key, value) in enumerate(b.items()):

					batch_index = int(key.split('_')[0])
					start_index = int(key.split('_')[1])
					end_index = int(key.split('_')[2])

					predict_barcodes = predict_barcodes + asap_object.adata.load_datainfo_batch(batch_index,start_index,end_index)

					if bi ==0 :
						asap_object.nmf['theta'] = value['theta']
						asap_object.nmf['corr'] = value['corr']
					else:
						asap_object.nmf['theta'] = np.vstack((asap_object.nmf['theta'],value['theta']))
						asap_object.nmf['corr'] = np.vstack((asap_object.nmf['corr'],value['corr']))
			
			asap_object.adata.obs = pd.DataFrame()
			asap_object.adata.obs['barcodes']= predict_barcodes
