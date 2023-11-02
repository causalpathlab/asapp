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
    if 'pbmc' in sample:
        ct = [ x.replace('@'+sample,'') for x in cl]
        return [ '-'.join(x.split('_')[1:]) for x in ct]
    elif 'sim' in sample:
        ct = [ x.replace('@'+sample,'') for x in cl]
        return [ '-'.join(x.split('_')[1:]) for x in ct]

    # elif 'pbmc' in sample:

    #     f='/data/sishir/data/pbmc_211k/pbmc_label.csv.gz'
    #     dfl = pd.read_csv(f)
    #     lmap = {x.split('@')[0]:y  for x,y in zip(dfl['cell'].values,dfl['celltype'].values)}
    #     cl = [x.split('@')[0] for x in cl]
    #     return [lmap[x] if x in lmap.keys() else 'others' for x in cl]

    # elif 'brca' in sample:
    #     f='/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_data/breastcancer/1_d_celltopic_label.csv.gz'
    #     dfl = pd.read_csv(f)
    #     dfl = dfl.loc[dfl['cell'].str.contains('GSE176078' ),:]
    #     dfl['cell'] = [ x.replace('_GSE176078','') for x in dfl['cell']]
    #     lmap = {x:y  for x,y in zip(dfl['cell'].values,dfl['celltype'].values)}
    #     return [lmap[x] if x in lmap.keys() else 'others' for x in cl]


######################################################

rho=1.0
sample = 'sim_r_'+str(rho)+'_d_10000_s_250_s_1_t_13_r_1.0'

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/examples/sim_noise_test/'
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

    # rp_data,rp_index = asappy.generate_randomprojection(asap_object,tree_depth=10)
    # rp = rp_data[0]['1_0_'+str(data_size)]['rp_data']
    # for d in rp_data[1:]:
    #     rp = np.vstack((rp,d[list(d.keys())[0]]['rp_data']))
    # rpi = np.array(rp_index[0])
    # for i in rp_index[1:]:rpi = np.hstack((rpi,i))
    # df=pd.DataFrame(rp)
    # df.index = rpi

    # from asappy.projection.rpstruct import projection_data
    # rp_mat_list = projection_data(10,asap_object.adata.uns['shape'][1])
    # mtx = asap_object.adata.load_data_batch(1,0,3250)
    # mtx = mtx.T


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
    asappy.plot_umap_df(dfumap,'celltype',asap_object.adata.uns['inpath']+'_rp_')



    asap_rp_s1,asap_rp_s2 =  calc_score([ str(x) for x in dfumap['celltype']],cluster)
    print('---RP-----')
    print('NMI:'+str(asap_rp_s1))
    print('ARI:'+str(asap_rp_s2))

    ## random projection plot
    # dfrp = df.copy()
    # dfrp.columns = ['rp'+str(x) for x in dfrp.columns]
    # dfrp.index = [x.split('@')[0] for x in dfrp.index.values]
    # dfrp['celltype'] = dfumap['celltype'].values
    # asappy.plot_randomproj(dfrp,'celltype',asap_object.adata.uns['inpath'])


def analyze_pseudobulk():
    asappy.generate_pseudobulk(asap_object,min_pseudobulk_size=100,tree_depth=10,normalize_pb='lscale')
    asappy.pbulk_cellcounthist(asap_object)

    cl = asap_object.adata.load_datainfo_batch(1,0,data_size)
    cl = [ x.replace('@'+sample,'') for x in cl]
    ct = celltype(cl,'sim') 
    pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
    asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])


def analyze_nmf():
    
    n_topics = 13
    cluster_resolution= 0.5
    
    # asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale',downsample_pseudobulk=False,pseudobulk_filter=False)
    asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale')
    asappy.asap_nmf(asap_object,num_factors=n_topics)
    
    asap_adata = asappy.generate_model(asap_object,return_object=True)
    asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
    
    
    print(asap_adata.obs.cluster.value_counts())
    asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)

    asappy.plot_umap(asap_adata,col='cluster')

    cl = asap_adata.obs.index.values
    cl = [ x.replace('@'+sample,'') for x in cl]
    ct = celltype(cl,'sim') 

    asap_adata.obs['celltype']  = pd.Categorical(ct)
    asappy.plot_umap(asap_adata,col='celltype')
    
    asap_s1,asap_s2 =  calc_score([str(x) for x in asap_adata.obs['celltype'].values],asap_adata.obs['cluster'].values)

    print('---NMF-----')
    print('NMI:'+str(asap_s1))
    print('ARI:'+str(asap_s2))

def generate_pbmc_sample():
    from scipy.sparse import save_npz
    from scipy.sparse import csr_matrix
    
    df = asap_object.adata.construct_batch_df(10000)
    cl = df.index.values
    cl = [ x.replace('@'+sample,'') for x in cl]
    ct = celltype(cl,'sim')
    df.index = [ str(x)+'_'+y for x,y in zip(range(df.shape[0]),ct) ] 
    df.index = [x.replace(' ','_') for x in df.index.values]

    sparse_matrix = csr_matrix(df.values)
    row_names = df.index
    col_names = df.columns

    with open('/data/sishir/database/pbmc_row_names.txt', 'w') as f: f.write("\n".join(row_names))
    with open('/data/sishir/database/pbmc_col_names.txt', 'w') as f: f.write("\n".join(col_names))
    save_npz('/data/sishir/database/pbmc_sparse_matrix.npz', sparse_matrix)

def noise_heatmap():
    df = asap_object.adata.construct_batch_df(3250)
    cl = [ x.replace('@'+sample,'') for x in df.index.values]
    ct = celltype(cl,'sim')
    sample_indxs = pd.DataFrame(ct).groupby(0).sample(frac=.1).index.values

    mtx = df.iloc[sample_indxs].to_numpy()



    # from asappy.preprocessing.hvgenes import get_gene_norm_var
    # genes_var = get_gene_norm_var(df.to_numpy())
    # hvg_percentile= 99.9
    # cutoff_percentile = np.percentile(genes_var, hvg_percentile)
    # print(cutoff_percentile,genes_var.min(),genes_var.mean(),genes_var.max())
    # genes_var_sel = np.where(genes_var > cutoff_percentile, 0, genes_var)
    # mtx = mtx[:,genes_var_sel!=0.0]
        
    mtx = np.log1p(mtx)
    mtx[mtx>1]=1

    df2 = pd.DataFrame(mtx)
    df2.index = [x.split('@')[0] for x in df.iloc[sample_indxs].index.values]
    df2.index = ['-'.join(x.split('_')[1:]) for x in df2.index.values]
    print(df2.shape)     
    
    import matplotlib.pylab as plt
    import seaborn as sns

    sns.heatmap(df2.T,cmap="Oranges")
    # sns.heatmap(np.log1p(mtx),cmap="Oranges")
    plt.savefig(wdir+'results/'+sample+'_hmap.png')
    plt.close()

noise_heatmap()

# analyze_randomproject()

# analyze_pseudobulk()

# analyze_nmf()

