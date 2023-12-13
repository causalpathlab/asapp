from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import asappy
import pandas as pd
import numpy as np


########################################

def calc_score(ct,cl):
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import adjusted_rand_score
    nmi =  normalized_mutual_info_score(ct,cl)
    ari = adjusted_rand_score(ct,cl)
    return nmi,ari


def celltype(cl,sample):
    ct = [ x.replace('@'+sample,'') for x in cl]
    return [ '-'.join(x.split('_')[1:]) for x in ct]


######################################################
import sys 
rho = float(sys.argv[1])
sample = 'sim_r_'+str(rho)+'_d_10000_s_250_s_1_t_13_r_0.1'

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_2_a_c/'
data_size = 25000
number_batches = 1


asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)


def analyze_randomproject():
    
    from asappy.clustering.leiden import leiden_cluster
    from umap.umap_ import find_ab_params, simplicial_set_embedding


    ### get random projection
    rp_data = asappy.generate_randomprojection(asap_object,tree_depth=10)
    rp_data = rp_data['full']
    rpi = asap_object.adata.load_datainfo_batch(1,0,rp_data.shape[0])
    df=pd.DataFrame(rp_data)
    df.index = rpi


    ### draw nearest neighbour graph and cluster
    snn,cluster = leiden_cluster(df.to_numpy(),resolution=1.0)
    pd.Series(cluster).value_counts() 



    ## umap cluster using neighbour graph
    min_dist = 0.5
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

    ct = celltype(dfumap['cell'].values,sample)
    dfumap['celltype'] = pd.Categorical(ct)
    asappy.plot_umap_df(dfumap,'celltype',asap_object.adata.uns['inpath']+'_rp_',pt_size=2.5,ftype='pdf')



    asap_rp_s1,asap_rp_s2 =  calc_score([ str(x) for x in dfumap['celltype']],cluster)
    print('---RP-----')
    print('NMI:'+str(asap_rp_s1))
    print('ARI:'+str(asap_rp_s2))


def analyze_pseudobulk():
    asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale')
    asappy.pbulk_cellcounthist(asap_object)

    cl = asap_object.adata.load_datainfo_batch(1,0,data_size)
    cl = [ x.replace('@'+sample,'') for x in cl]
    ct = celltype(cl,'sim') 
    pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
    asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])


def analyze_nmf():
    
    n_topics = 13
    cluster_resolution= 0.5
    
    asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale')
    asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
    
    asap_adata = asappy.generate_model(asap_object,return_object=True)
    asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
    
    
    print(asap_adata.obs.cluster.value_counts())
    asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.5)

    asappy.plot_umap(asap_adata,col='cluster',pt_size=2.5,ftype='pdf')

    cl = asap_adata.obs.index.values
    cl = [ x.replace('@'+sample,'') for x in cl]
    ct = celltype(cl,'sim') 

    asap_adata.obs['celltype']  = pd.Categorical(ct)
    asappy.plot_umap(asap_adata,col='celltype',pt_size=2.5,ftype='pdf')
    
    asap_s1,asap_s2 =  calc_score([str(x) for x in asap_adata.obs['celltype'].values],asap_adata.obs['cluster'].values)

    print('---NMF-----')
    print('NMI:'+str(asap_s1))
    print('ARI:'+str(asap_s2))

def noise_heatmap():
    df = asap_object.adata.construct_batch_df(3250)
    cl = [ x.replace('@'+sample,'') for x in df.index.values]
    ct = celltype(cl,'sim')
    sample_indxs = pd.DataFrame(ct).groupby(0).sample(frac=.03).index.values

    # mtx = df.iloc[sample_indxs].to_numpy()
    # mtx = np.log1p(mtx)
    # mtx[mtx>1]=1


    from anndata import AnnData
    import scanpy as sc
    import numpy as np

    df = df.iloc[sample_indxs]
    adata = AnnData(df.to_numpy())
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata,n_top_genes=200)
    adata = adata[:, adata.var.highly_variable]
    mtx = adata.to_df().to_numpy()
    
    df2 = pd.DataFrame(mtx)
    df2.index = [x.split('@')[0] for x in df.index.values]
    df2.index = ['-'.join(x.split('_')[1:]) for x in df2.index.values]
    print(df2.shape)     
    
    import matplotlib.pylab as plt
    import seaborn as sns
    from matplotlib.pyplot import figure
    w=10
    h=15
    figure(figsize=(h, w), dpi=300)

    # sns.heatmap(df2.T,cmap="viridis")
    
    ## for 1.0 case cluster 
    sns.clustermap(df2.T,cmap="Oranges",col_cluster=False)
    
    plt.savefig(wdir+'results/'+sample+'_hmap.png',dpi=600)
    plt.savefig(wdir+'results/'+sample+'_hmap.pdf',dpi=600)
    plt.close()

noise_heatmap()

# analyze_randomproject()

# analyze_pseudobulk()

# analyze_nmf()
 