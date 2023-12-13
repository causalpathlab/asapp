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

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_4_a/'

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

# def extra(): 
#     asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')

#     #### cells by factor plot 

#     pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
#     df = pd.DataFrame(pmf2t['prop'])
#     df.columns = ['t'+str(x) for x in df.columns]


#     df.reset_index(inplace=True)
#     cl = asap_adata.obs.index.values
#     ct = ct = getct(cl,sample)
#     df['celltype'] = ct

#     def sample_n_rows(group):
#         return group.sample(n=min(n, len(group)))

#     n= 3125
#     sampled_df = df.groupby('celltype', group_keys=False).apply(sample_n_rows)
#     sampled_df.reset_index(drop=True, inplace=True)
#     print(sampled_df)

#     sampled_df.drop(columns='index',inplace=True)
#     sampled_df.set_index('celltype',inplace=True)

#     import matplotlib.pylab as plt
#     import seaborn as sns

#     sns.clustermap(sampled_df,cmap='viridis',col_cluster=False)
#     plt.savefig(asap_adata.uns['inpath']+'_prop_hmap.png');plt.close()



#     ###################


#     dfm = pd.melt(sampled_df,id_vars=['index','celltype'])
#     dfm.columns = ['id','celltype','topic','value']

#     dfm['id'] = pd.Categorical(dfm['id'])
#     dfm['celltype'] = pd.Categorical(dfm['celltype'])
#     dfm['topic'] = pd.Categorical(dfm['topic'])

#     custom_palette = [
#     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
#     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
#     "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
#     "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
#     '#6b6ecf', '#8ca252',  '#8c6d31', '#bd9e39', '#d6616b'
#     ]

#     p = (ggplot(data=dfm, mapping=aes(x='id', y='value', fill='topic')) +
#         geom_bar(position="stack", stat="identity", size=0) +
#         scale_fill_manual(values=custom_palette) +
#         facet_grid('~ celltype', scales='free', space='free'))

#     p = p + theme(
#         plot_background=element_rect(fill='white'),
#         panel_background = element_rect(fill='white'),
#         axis_text_x=element_blank())
#     p.save(filename = asap_adata.uns['inpath']+'_sc_topic_struct.png', height=5, width=15, units ='in', dpi=300)


#     pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
#     df = pd.DataFrame(asap_adata.obsm['corr'])
#     df.columns = ['t'+str(x) for x in df.columns]
#     clusts = [x for x in df.idxmax(axis=1)]

#     from asappy.clustering.leiden import refine_clusts

#     clusts = refine_clusts(df.to_numpy(),clusts,25)
#     dfct = pd.DataFrame(clusts,columns=['topic'])
#     dfct['celltype'] = ct
#     dfct['score'] = 1

#     ######### celltype proportion per topic
#     dfct_grp = dfct.groupby(['topic','celltype']).count().reset_index()
#     celltopic_sum = dict(dfct.groupby(['topic'])['score'].sum())
#     dfct_grp['ncount'] = [x/celltopic_sum[y] for x,y in zip(dfct_grp['score'],dfct_grp['topic'])]


#     custom_palette = [
#     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
#     "#8c564b", "#e377c2", "#7f7f7f"]


#     p = (ggplot(data=dfct_grp, mapping=aes(x='topic', y='ncount', fill='celltype')) +
#         geom_bar(position="stack", stat="identity", size=0) +
#         scale_fill_manual(values=custom_palette) )

#     p = p + theme(
#         plot_background=element_rect(fill='white'),
#         panel_background = element_rect(fill='white'))
#         # axis_text_x=element_blank())
#     p.save(filename = asap_adata.uns['inpath']+'_topic_ctprop.pdf', height=5, width=15, units ='in', dpi=600)



#     ############# topic propertion per cell type
#     dfct_grp = dfct.groupby(['celltype','topic']).count().reset_index()


#     celltopic_sum = dict(dfct.groupby(['celltype'])['score'].sum())
#     dfct_grp['ncount'] = [x/celltopic_sum[y] for x,y in zip(dfct_grp['score'],dfct_grp['celltype'])]


#     custom_palette = [
#     "#b3446c", "#dcd300", "#882d17",
#     "#8db600","#654522","#e25822","#2b3d26","#006FA6","blue"]

#     # "orange", "#1CE6FF", "fuchsia", "#7A4900", "green","gray","#006FA6" ,"limegreen", "red", "blue"]

#     p = (ggplot(data=dfct_grp, mapping=aes(x='celltype', y='ncount', fill='topic')) +
#         geom_bar(position="stack", stat="identity", size=0) +
#         scale_fill_manual(values=custom_palette) )

#     p = p + theme(
#         plot_background=element_rect(fill='white'),
#         panel_background = element_rect(fill='white'))
#         # axis_text_x=element_blank())
#     p.save(filename = asap_adata.uns['inpath']+'_ct_topicprop.pdf', height=5, width=15, units ='in', dpi=600)




#     def sample_n_rows(group):
#         return group.sample(n=min(n, len(group)))

#     n=25
#     sampled_df = df.groupby('celltype', group_keys=False).apply(sample_n_rows)
#     sampled_df.reset_index(drop=True, inplace=True)
#     print(sampled_df)


#     dfm = pd.melt(sampled_df,id_vars=['index','celltype'])
#     dfm.columns = ['id','celltype','topic','value']

#     dfm['id'] = pd.Categorical(dfm['id'])
#     dfm['celltype'] = pd.Categorical(dfm['celltype'])
#     dfm['topic'] = pd.Categorical(dfm['topic'])

#     custom_palette = [
#     "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
#     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
#     "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
#     "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
#     '#6b6ecf', '#8ca252',  '#8c6d31', '#bd9e39', '#d6616b'
#     ]

#     p = (ggplot(data=dfm, mapping=aes(x='id', y='value', fill='topic')) +
#         geom_bar(position="stack", stat="identity", size=0) +
#         scale_fill_manual(values=custom_palette) +
#         facet_grid('~ celltype', scales='free', space='free'))

#     p = p + theme(
#         plot_background=element_rect(fill='white'),
#         panel_background = element_rect(fill='white'),
#         axis_text_x=element_blank())
#     p.save(filename = asap_adata.uns['inpath']+'_sc_topic_struct_corr.png', height=5, width=15, units ='in', dpi=300)

