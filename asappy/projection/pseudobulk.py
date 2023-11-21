import threading
import queue
import numpy as np
import pandas as pd

from .rpstruct import projection_data, get_pseudobulk, get_randomprojection
import logging
logger = logging.getLogger(__name__)

def generate_random_projection_data(var_dims,tree_depth):
    return projection_data(tree_depth,var_dims)

def generate_randomprojection_batch(asap_object,batch_i,start_index,end_index,rp_mat_list,result_queue,result_indxes_queue,lock,sema):

    if batch_i <= asap_object.adata.uns['number_batches']:

        logging.info('Random projection generation for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index))
        
        sema.acquire()

        lock.acquire()
        local_mtx = asap_object.adata.load_data_batch(batch_i,start_index,end_index)	
        local_mtx_indxes = asap_object.adata.load_datainfo_batch(batch_i,start_index,end_index)
        result_indxes_queue.put(local_mtx_indxes)	
        lock.release()

        get_randomprojection(local_mtx.T, 
            rp_mat_list, 
            str(batch_i) +'_' +str(start_index)+'_'+str(end_index),
            result_queue
            )
        sema.release()			
    else:
        logging.info('Random projection NOT generated for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index)+ ' '+str(batch_i) + ' > ' +str(asap_object.adata.uns['number_batches']))

def generate_pseudobulk_batch(asap_object,batch_i,start_index,end_index,rp_mat_list,normalize_raw,normalize_pb,pb_aggregation,hvg_selection,gene_mean_z,gene_var_z,result_queue,lock,sema):

    if batch_i <= asap_object.adata.uns['number_batches']:

        logging.info('Pseudo-bulk generation for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index))
        
        sema.acquire()

        lock.acquire()
        local_mtx = asap_object.adata.load_data_batch(batch_i,start_index,end_index)	
        lock.release()

        get_pseudobulk(local_mtx.T, 
            rp_mat_list, 
            asap_object.adata.uns['downsample_pseudobulk'],
            asap_object.adata.uns['downsample_size'],
            pb_aggregation,
            str(batch_i) +'_' +str(start_index)+'_'+str(end_index),
            normalize_raw,
            normalize_pb,
            hvg_selection,
            gene_mean_z,
            gene_var_z,
            result_queue
            )
        sema.release()			
    # else:
    #     logging.info('Pseudo-bulk NOT generated for '+str(batch_i) +'_' +str(start_index)+'_'+str(end_index)+ ' '+str(batch_i) + ' > ' +str(asap_object.adata.uns['number_batches']))

def filter_pseudobulk(asap_object,pseudobulk_result,min_size):
    
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
        asap_object.adata.uns['pseudobulk']['pb_hvgs'] = pseudobulk_result['full']['pb_hvgs'] 

    else:
        asap_object.adata.uns['pseudobulk']['pb_map'] = {}
        for indx,result_batch in enumerate(pseudobulk_result):

            pseudobulk_map = result_batch[[k for k in result_batch.keys()][0]]['pb_map']
            pb = result_batch[[k for k in result_batch.keys()][0]]['pb_data']
            
            hvgs = result_batch[[k for k in result_batch.keys()][0]]['pb_hvgs']
            
            sample_counts = np.array([len(pseudobulk_map[x])for x in pseudobulk_map.keys()])
            keep_indices = np.where(sample_counts>min_size)[0].flatten() 

            pb = pb[:,keep_indices]
            pseudobulk_map = {key: value for i, (key, value) in enumerate(pseudobulk_map.items()) if i in keep_indices}

            if indx == 0:
                asap_object.adata.uns['pseudobulk']['pb_data'] = pb
                asap_object.adata.uns['pseudobulk']['pb_hvgs'] = hvgs
            else:
                
                ### need to correct hvgs index from multiple batches
                prv_hvgs = asap_object.adata.uns['pseudobulk']['pb_hvgs']
                combine_hvgs = np.array([a or b for a,b in zip(prv_hvgs,hvgs)])
                
                prv_pbdata = asap_object.adata.uns['pseudobulk']['pb_data']
                
                updated_prv_pbdata = pd.DataFrame(np.zeros((len(combine_hvgs), prv_pbdata.shape[1])))
                updated_prv_pbdata.loc[prv_hvgs] = prv_pbdata
                
                updated_current_pbdata = pd.DataFrame(np.zeros((len(combine_hvgs), pb.shape[1])))
                updated_current_pbdata.loc[hvgs] = pb
                
                updated_current_pbdata = updated_current_pbdata[combine_hvgs]
                updated_prv_pbdata = updated_prv_pbdata[combine_hvgs]
                
                asap_object.adata.uns['pseudobulk']['pb_data'] = np.hstack((updated_prv_pbdata,updated_current_pbdata))
                
                asap_object.adata.uns['pseudobulk']['pb_hvgs'] = combine_hvgs
            
            asap_object.adata.uns['pseudobulk']['pb_map'][[k for k in result_batch.keys()][0]] = pseudobulk_map

    logging.info('Pseudo-bulk size :' +str(asap_object.adata.uns['pseudobulk']['pb_data'].shape))

def generate_randomprojection(asap_object,tree_depth,maxthreads=16):
    asap_object.adata.uns['tree_depth'] = tree_depth

    logging.info('Random projection generation... \n'+
        'tree_depth : ' + str(tree_depth)+'\n'
        )

    total_cells = asap_object.adata.uns['shape'][0]
    total_genes = asap_object.adata.uns['shape'][1]
    batch_size = asap_object.adata.uns['batch_size']

    logging.info('Data size...cell x gene '+str(total_cells) +'x'+ str(total_genes))
    logging.info('Batch size... '+str(batch_size))
    logging.info('Data batch to process... '+str(asap_object.adata.uns['number_batches']))

    rp_mat_list = generate_random_projection_data(asap_object.adata.uns['shape'][1],asap_object.adata.uns['tree_depth'])
    
    if total_cells<batch_size:

        rp_data = get_randomprojection(asap_object.adata.X.T, rp_mat_list,'full')
        return rp_data

    else:

        threads = []
        result_queue = queue.Queue()
        result_indxes_queue = queue.Queue()
        lock = threading.Lock()
        sema = threading.Semaphore(value=maxthreads)

        for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

            iend = min(istart + batch_size, total_cells)
                            
            thread = threading.Thread(target=generate_randomprojection_batch, args=(asap_object,i,istart,iend, rp_mat_list,result_queue,result_indxes_queue,lock,sema))
            
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

        rp_data = []
        rp_data_indxes = []
        while not result_queue.empty():
            rp_data.append(result_queue.get())
            rp_data_indxes.append(result_indxes_queue.get())
    
        return rp_data,rp_data_indxes
 
def generate_pseudobulk(
    asap_object,
    tree_depth,
    normalize_raw=None,
    normalize_pb=None,
    pb_aggregation = 'SVD',
    hvg_selection=False,
    gene_mean_z=10,
    gene_var_z=2,
    downsample_pseudobulk=True,
    downsample_size=100,
    maxthreads=16,
    pseudobulk_filter=True,
    pseudobulk_filter_size=5
    ):
    
    asap_object.adata.uns['tree_depth'] = tree_depth
    asap_object.adata.uns['downsample_pseudobulk'] = downsample_pseudobulk
    asap_object.adata.uns['downsample_size'] = downsample_size
    
    logging.info('Pseudo-bulk generation... \n'+
        'tree_depth : ' + str(tree_depth)+'\n'+
        'downsample_pseudobulk :' + str(downsample_pseudobulk)+'\n'+
        'downsample_size :' + str(downsample_size)+'\n'+
        'pseudobulk_filter_size :' + str(pseudobulk_filter_size)+'\n'
        )

    total_cells = asap_object.adata.uns['shape'][0]
    total_genes = asap_object.adata.uns['shape'][1]
    batch_size = asap_object.adata.uns['batch_size']

    logging.info('Data size...cell x gene '+str(total_cells) +'x'+ str(total_genes))
    logging.info('Batch size... '+str(batch_size))
    logging.info('Data batch to process... '+str(asap_object.adata.uns['number_batches']))

    rp_mat_list = generate_random_projection_data(asap_object.adata.uns['shape'][1],asap_object.adata.uns['tree_depth'])
    
    if total_cells<batch_size:

        pseudobulk_result = get_pseudobulk(asap_object.adata.X.T, rp_mat_list,asap_object.adata.uns['downsample_pseudobulk'],asap_object.adata.uns['downsample_size'],pb_aggregation,'full',normalize_raw,normalize_pb,hvg_selection,gene_mean_z,gene_var_z)

    else:

        threads = []
        result_queue = queue.Queue()
        lock = threading.Lock()
        sema = threading.Semaphore(value=maxthreads)

        for (i, istart) in enumerate(range(0, total_cells,batch_size), 1): 

            iend = min(istart + batch_size, total_cells)
                            
            thread = threading.Thread(target=generate_pseudobulk_batch, args=(asap_object,i,istart,iend, rp_mat_list,normalize_raw,normalize_pb,pb_aggregation,hvg_selection,gene_mean_z,gene_var_z,result_queue,lock,sema))
            
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

        pseudobulk_result = []
        while not result_queue.empty():
            pseudobulk_result.append(result_queue.get())
    
    pb_min_size = 0
    if pseudobulk_filter:
        pb_min_size = pseudobulk_filter_size

    filter_pseudobulk(asap_object,pseudobulk_result,pb_min_size)
    
    
'''
1. dont create pseudobulk after you run pca for entire dataset
2. we can keep buffer size 
3. down sampling - sub sampling cells in the pseudobulk -> benefit is to create a homogeneous pseudobulk samples
'''