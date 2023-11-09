######################################################
##### single cell nmf
######################################################
import asappy
import sys


sample ='pbmc_s_10000_t_10_r_0.1_s_1'
data_size = 10000
n_topics = 10
cluster_resolution = 0.1
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/examples/pbmc/'


# asappy.create_asap_data(sample=sample,working_dirpath=wdir)

number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)

# asappy.generate_pseudobulk(asap_object,tree_depth=10)
# asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_raw='unitnorm')
# asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale')
asappy.generate_pseudobulk(asap_object,tree_depth=10,hvg_selection=True,gene_var_z=1)

asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
asappy.generate_model(asap_object)

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from collections import Counter
import pandas as pd

def getct(ids,sample):
    
    dfid = pd.DataFrame(ids,columns=['cell'])
    dfl = pd.read_csv(wdir+'results/'+sample.split('_')[0]+'_label.csv.gz')
    dfjoin = pd.merge(dfl,dfid,on='cell',how='right')
    ct = dfjoin['celltype'].values
        
    return ct

def calculate_purity(true_labels, cluster_labels):
    cluster_set = set(cluster_labels)
    total_correct = sum(max(Counter(true_labels[i] for i, cl in enumerate(cluster_labels) if cl == cluster).values()) 
                        for cluster in cluster_set)
    return total_correct / len(true_labels)

def calc_score(ct,cl,sample):
    ct = getct(ct,sample)    
    nmi =  normalized_mutual_info_score(ct,cl)
    ari =  adjusted_rand_score(ct,cl)
    purity = calculate_purity(ct,cl)
    return nmi,ari,purity


def analyze_rp():
    import pandas as pd
    import numpy as np
    from asappy.clustering.leiden import leiden_cluster
    from umap.umap_ import find_ab_params, simplicial_set_embedding


    ### get random projection
    rp_data,rp_index = asappy.generate_randomprojection(asap_object,tree_depth=10)
    rp = rp_data[0]['1_0_'+str(data_size)]['rp_data']
    for d in rp_data[1:]:
        rp = np.vstack((rp,d[list(d.keys())[0]]['rp_data']))
    rpi = np.array(rp_index[0])
    for i in rp_index[1:]:rpi = np.hstack((rpi,i))
    df=pd.DataFrame(rp)
    df.index = rpi

    ### draw nearest neighbour graph and cluster
    snn,cluster = leiden_cluster(df.to_numpy(),resolution=0.1)
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

    ids = [ x.replace('@'+sample,'') for x in dfumap.cell.values]
    ct = getct(ids,sample)
    dfumap['celltype'] = pd.Categorical(ct)
    asappy.plot_umap_df(dfumap,'celltype',asap_object.adata.uns['inpath']+'_rp_')

    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import adjusted_rand_score

    def calc_score(ct,cl):
        nmi =  normalized_mutual_info_score(ct,cl)
        ari = adjusted_rand_score(ct,cl)
        return nmi,ari

    asap_rp_s1,asap_rp_s2 =  calc_score([ str(x) for x in dfumap['celltype']],cluster)

    print('NMI:'+str(asap_rp_s1))
    print('ARI:'+str(asap_rp_s2))
    
    
def analyze_pseudobulk(asap_object):
    asappy.pbulk_cellcounthist(asap_object)

    cl = asap_object.adata.load_datainfo_batch(1,0,data_size)
    cl = [ x.replace('@'+sample,'') for x in cl]
    ct = getct(cl,sample) 
    pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
    asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])


def run_nmf():
    import asappy
    import anndata as an
    from pyensembl import ensembl_grch38

    asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')

    ##### cluster and celltype umap
    asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)

    # print(asap_adata.obs.cluster.value_counts())
    asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
    asappy.plot_umap(asap_adata,col='cluster')
    ids = [ x.replace('@'+ sample,'') for x in asap_adata.obs.index.values]
    asap_adata.obs['celltype'] = getct(ids,sample)
    asappy.plot_umap(asap_adata,col='celltype')

    # asap_adata.write(wdir+'results/'+sample+'.h5asapad')

    # asap = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')
   
    asap_s1,asap_s2,asap_s3 =  calc_score(ids,asap_adata.obs['cluster'].values,sample)
    print(asap_s1,asap_s2,asap_s3)


analyze_pseudobulk(asap_object)
run_nmf()