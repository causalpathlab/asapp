import threading
import queue
import numpy as np

from .rpstruct import projection_data, get_pseudobulk
import logging
logger = logging.getLogger(__name__)

def generate_random_projection_data(var_dims,tree_depth):
    return projection_data(tree_depth,var_dims)

def generate_pseudobulk_batch(asap_object,batch_i,start_index,end_index,rp_mat,result_queue,lock,sema):

    if batch_i <= asap_object.adata.uns['number_batches']:

        logging.info('Pseudo-bulk generation for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index))
        
        sema.acquire()

        lock.acquire()
        local_mtx = asap_object.adata.load_data_batch(batch_i,start_index,end_index)	
        lock.release()

        get_pseudobulk(local_mtx.T, 
            rp_mat, 
            asap_object.adata.uns['downsample_pseudobulk'],asap_object.adata.uns['downsample_size'],
            str(batch_i) +'_' +str(start_index)+'_'+str(end_index),
            result_queue
            )
        sema.release()			
    else:
        logging.info('Pseudo-bulk NOT generated for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index)+ ' '+str(batch_i) + ' > ' +str(asap_object.adata.uns['number_batches']))

def filter_pseudobulk(asap_object,pseudobulk_result,min_size=5):
    
    asap_object.adata.uns['pseudobulk'] = {}

    logging.info('Pseudo-bulk sample filtering...')

    if len(pseudobulk_result) == 1 and asap_object.adata.uns['run_full_data']:
        
        pseudobulk_map = pseudobulk_result['full']['pb_map'] 

        sample_counts = np.array([len(pseudobulk_map[x])for x in pseudobulk_map.keys()])
        keep_indices = np.where(sample_counts>min_size)[0].flatten() 

        asap_object.adata.uns['pseudobulk']['pb_data'] = pseudobulk_result['full']['pb_data'][:,keep_indices]
        pseudobulk_indices = {key: value for i, (key, value) in enumerate(pseudobulk_map.items()) if i in keep_indices}
        batch_index = str(1)+'_'+str(0)+'_'+str(asap_object.adata.uns['shape'][0])
        asap_object.adata.uns['pseudobulk']['pb_map'] = {batch_index:pseudobulk_indices}

    else:
        asap_object.adata.uns['pseudobulk']['pb_map'] = {}
        for indx,result_batch in enumerate(pseudobulk_result):

            pseudobulk_map = result_batch[[k for k in result_batch.keys()][0]]['pb_map']
            pb = result_batch[[k for k in result_batch.keys()][0]]['pb_data']

            sample_counts = np.array([len(pseudobulk_map[x])for x in pseudobulk_map.keys()])
            keep_indices = np.where(sample_counts>min_size)[0].flatten() 

            pb = pb[:,keep_indices]
            pseudobulk_map = {key: value for i, (key, value) in enumerate(pseudobulk_map.items()) if i in keep_indices}

            if indx == 0:
                asap_object.adata.uns['pseudobulk']['pb_data'] = pb
            else:
                asap_object.adata.uns['pseudobulk']['pb_data'] = np.hstack((asap_object.adata.uns['pseudobulk']['pb_data'],pb))
            
            asap_object.adata.uns['pseudobulk']['pb_map'][[k for k in result_batch.keys()][0]] = pseudobulk_map

    logging.info('Pseudo-bulk size :' +str(asap_object.adata.uns['pseudobulk']['pb_data'].shape))

def generate_pseudobulk(asap_object,tree_depth,downsample_pseudobulk=True,downsample_size=100,maxthreads=16,pseudobulk_filter_size=5):
    asap_object.adata.uns['tree_depth'] = tree_depth
    asap_object.adata.uns['downsample_pseudobulk'] = downsample_pseudobulk
    asap_object.adata.uns['downsample_size'] = downsample_size
    
    logging.info('Pseudo-bulk generation...')
    
    total_cells = asap_object.adata.uns['shape'][0]
    total_genes = asap_object.adata.uns['shape'][1]
    batch_size = asap_object.adata.uns['batch_size']

    logging.info('Data size...cell x gene '+str(total_cells) +'x'+ str(total_genes))
    logging.info('Batch size... '+str(batch_size))
    logging.info('Data batch to process... '+str(asap_object.adata.uns['number_batches']))

    rp_mat = generate_random_projection_data(asap_object.adata.uns['shape'][1],asap_object.adata.uns['tree_depth'])
    
    if total_cells<batch_size:

        pseudobulk_result = get_pseudobulk(asap_object.adata.X.T, rp_mat,asap_object.adata.uns['downsample_pseudobulk'],asap_object.adata.uns['downsample_size'],'full')

    else:

        threads = []
        result_queue = queue.Queue()
        lock = threading.Lock()
        sema = threading.Semaphore(value=maxthreads)

        for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

            iend = min(istart + batch_size, total_cells)
                            
            thread = threading.Thread(target=generate_pseudobulk_batch, args=(asap_object,i,istart,iend, rp_mat,result_queue,lock,sema))
            
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

        pseudobulk_result = []
        while not result_queue.empty():
            pseudobulk_result.append(result_queue.get())
    
    filter_pseudobulk(asap_object,pseudobulk_result,pseudobulk_filter_size)
        