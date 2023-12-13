from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import asappy
import pandas as pd
import numpy as np
import anndata as an

########################################

def calc_score(ct,cl):
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import adjusted_rand_score
    nmi =  normalized_mutual_info_score(ct,cl)
    ari = adjusted_rand_score(ct,cl)
    return nmi,ari


def getct(ids,sample):
    sample='pbmc'
    ids = [x.replace('@'+sample,'') for x in ids]
    dfid = pd.DataFrame(ids,columns=['cell'])
    dfl = pd.read_csv(wdir+'results/'+sample+'_celltype.csv.gz')
    dfl['cell'] = [x.replace('@'+sample,'') for x in dfl['cell']]
    dfjoin = pd.merge(dfl,dfid,on='cell',how='right')
    ct = dfjoin['celltype'].values
    
    return ct

######################################################
import sys 
sample = 'pbmctcell'

wdir = 'experiments/asapp/figures/fig_4_a/'

data_size = 90000
number_batches = 1


# asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)


def analyze_randomproject():
    
    from asappy.clustering.leiden import leiden_cluster,kmeans_cluster
    from umap.umap_ import find_ab_params, simplicial_set_embedding


    ### get random projection
    # rp_data,rp_index = asappy.generate_randomprojection(asap_object,tree_depth=10)
    # rp = rp_data[0]['1_0_100000']['rp_data']
    # for d in rp_data[1:]:
    #     rp = np.vstack((rp,d[list(d.keys())[0]]['rp_data']))
    # rpi = np.array(rp_index[0])
    # for i in rp_index[1:]:rpi = np.hstack((rpi,i))
    # df=pd.DataFrame(rp)
    # df.index = rpi

    rp_data = asappy.generate_randomprojection(asap_object,tree_depth=10)
    rp_data = rp_data['full']
    rpi = asap_object.adata.load_datainfo_batch(1,0,rp_data.shape[0])
    df=pd.DataFrame(rp_data)
    df.index = rpi

    ### draw nearest neighbour graph and cluster
    res = 1.0
    umapdist = 0.1
    
    
    snn,cluster = leiden_cluster(df.to_numpy(),resolution=res)
    pd.Series(cluster).value_counts() 



    ## umap cluster using neighbour graph
    min_dist = umapdist
    n_components = 2
    spread: float = 1.0
    alpha: float = 1.0
    gamma: float = 1.0
    negative_sample_rate: int = 5
    maxiter = None
    default_epochs = 500 if snn.shape[0] <= 10000 else 200
    n_epochs = default_epochs if maxiter is None else maxiter
    random_state = np.random.RandomState(42)

    a, b = find_ab_params(spread, min_dist)

    umap_coords = simplicial_set_embedding(
    data = df.to_numpy(),
    graph = snn,
    n_components=n_components,
    initial_alpha = alpha,
    a = a,
    b = b,
    gamma = gamma,
    negative_sample_rate = negative_sample_rate,
    n_epochs = n_epochs,
    init='spectral',
    random_state = random_state,
    metric = 'cosine',
    metric_kwds = {},
    densmap=False,
    densmap_kwds={},
    output_dens=False
    )

    dfumap = pd.DataFrame(umap_coords[0])
    dfumap['cell'] = rpi
    dfumap.columns = ['umap1','umap2','cell']


    ct = getct(dfumap['cell'].values,sample)
    dfumap['celltype'] = pd.Categorical(ct)
    asappy.plot_umap_df(dfumap,'celltype',asap_object.adata.uns['inpath']+'_rp_',pt_size=0.5,ftype='png')



    asap_rp_s1,asap_rp_s2 =  calc_score([ str(x) for x in dfumap['celltype']],cluster)
    print('---RP-----')
    print('NMI:'+str(asap_rp_s1))
    print('ARI:'+str(asap_rp_s2))


def analyze_pseudobulk():
    asappy.generate_pseudobulk(asap_object,tree_depth=10)
    asappy.pbulk_cellcounthist(asap_object)

    cl = asap_object.adata.load_datainfo_batch(1,0,asap_object.adata.uns['shape'][0])
    ct = getct(cl,sample)
    pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
    asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])

    
def analyze_nmf():

    asappy.generate_pseudobulk(asap_object,tree_depth=10)
    
    n_topics = 50 ## paper 
    asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
    
    # asap_adata = asappy.generate_model(asap_object,return_object=True)
    asappy.generate_model(asap_object)
    
    asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')
    
    cluster_resolution= 1.0 ## paper
    asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
    print(asap_adata.obs.cluster.value_counts())
    
    ## min distance 0.5 paper
    asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.01)
    asappy.plot_umap(asap_adata,col='cluster',pt_size=0.5,ftype='png')

    cl = asap_adata.obs.index.values
    ct = getct(cl,sample)
    asap_adata.obs['celltype']  = pd.Categorical(ct)
    asappy.plot_umap(asap_adata,col='celltype',pt_size=0.5,ftype='png')


    import h5py as hf
    f = hf.File('/data/sishir/data/pbmc_211k/pbmc_multimodal.h5seurat','r')
    ct2 = [x.decode('utf-8') for x in f['meta.data']['celltype.l2']]
    cid = [x.decode('utf-8') for x in f['cell.names']]
    dfl = pd.DataFrame([cid,ct2]).T
    dfl.columns = ['cell','celltype']
    
    dfid = pd.DataFrame(cl,columns=['cell'])
    dfid['cell'] = [x.replace('@'+sample,'') for x in dfid['cell']]
    dfjoin = pd.merge(dfl,dfid,on='cell',how='right')
    ct2 = dfjoin['celltype'].values
    asap_adata.obs['celltypel2']  = pd.Categorical(ct2)
    asappy.plot_umap(asap_adata,col='celltypel2',pt_size=0.5,ftype='png')

    
    asap_adata.write(wdir+'results/'+sample+'.h5asapad')
    
    ## top 10 main paper
    asappy.plot_gene_loading(asap_adata,top_n=10,max_thresh=25)
    
    asap_s1,asap_s2 =  calc_score([str(x) for x in asap_adata.obs['celltype'].values],asap_adata.obs['cluster'].values)

    print('---NMF-----')
    print('NMI:'+str(asap_s1))
    print('ARI:'+str(asap_s2))

# analyze_randomproject()

# analyze_pseudobulk()

analyze_nmf()
