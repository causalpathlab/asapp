from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import asappy
import pandas as pd
import numpy as np
import anndata as an

from asappy.plotting.palette import get_colors
########################################

def calc_score(ct,cl):
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics.cluster import adjusted_rand_score
    nmi =  normalized_mutual_info_score(ct,cl)
    ari = adjusted_rand_score(ct,cl)
    return nmi,ari


def getct(ids,sample):
    ids = [x.replace('@'+sample,'') for x in ids]
    dfid = pd.DataFrame(ids,columns=['cell'])
    dfl = pd.read_csv(wdir+'results/pbmc_celltype.csv.gz')
    dfl['cell'] = [x.replace('@'+sample,'') for x in dfl['cell']]
    dfjoin = pd.merge(dfl,dfid,on='cell',how='right')
    ct = dfjoin['celltype'].values
    
    return ct

######################################################
import sys 
sample = 'pbmcprot'

wdir = 'experiments/asapp/figures/fig_4_a/'

data_size = 90000
number_batches = 2


# asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)


def analyze_randomproject():
    
    from asappy.clustering.leiden import leiden_cluster,kmeans_cluster
    from umap.umap_ import find_ab_params, simplicial_set_embedding


    ## get random projection
    rp_data,rp_index = asappy.generate_randomprojection(asap_object,tree_depth=10)
    rp = rp_data[0]['1_0_90000']['rp_data']
    for d in rp_data[1:]:
        rp = np.vstack((rp,d[list(d.keys())[0]]['rp_data']))
    rpi = np.array(rp_index[0])
    for i in rp_index[1:]:rpi = np.hstack((rpi,i))
    df=pd.DataFrame(rp)
    df.index = rpi
    
    cluster = kmeans_cluster(df.to_numpy(),k=161)
	pd.Series(cluster).value_counts() 
 
	df.to_csv(wdir+'results/pbmcprot_rp.csv.gz',compression='gzip')
	pd.DataFrame(cluster,columns=['cluster']).to_csv(wdir+'results/tabsap_rp_kmeanscluster.csv.gz',compression='gzip')

	
	umap_2d = UMAP(n_components=2, init='random', random_state=0,min_dist=0.1)

	proj_2d = umap_2d.fit_transform(df.to_numpy())


	dfumap = pd.DataFrame(proj_2d[:,])
	dfumap['cell'] = rpi
	dfumap.columns = ['umap1','umap2','cell']

	ct = getct(dfumap['cell'].values,sample)
	dfumap['celltype'] = pd.Categorical(ct)
	dfumap['cluster'] = cluster
 

	pt_size = 0.1
	gradient_palette = get_colors(dfumap['celltype'].nunique())
 
	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_fill_manual(values=gradient_palette) 
		# guides(color=guide_legend(override_aes={'size': legend_size}))
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
  		legend_position='none')
 
	fname = asap_object.adata.uns['inpath']+'_rp_celltype_'+'umap.png'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)

	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='cluster')) +
		geom_point(size=pt_size) +
		scale_fill_manual(values=gradient_palette) 
		# guides(color=guide_legend(override_aes={'size': legend_size}))
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
  		legend_position='none')
 
	fname = asap_object.adata.uns['inpath']+'_rp_cluster_'+'umap.png'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)
 
	dfumap.to_csv(wdir+'results/tabsap_rp_dfumap.csv.gz',compression='gzip')
	
	''' 
	asap_rp_s1,asap_rp_s2 =  calc_score([ str(x) for x in dfumap['celltype']],cluster)
	print('---RP-----')
	print('NMI:'+str(asap_rp_s1))
	print('ARI:'+str(asap_rp_s2))
	NMI:0.2659732572314774
	ARI:0.05683899341197782
	'''

def analyze_pseudobulk():
    asappy.generate_pseudobulk(asap_object,tree_depth=10)
    asappy.pbulk_cellcounthist(asap_object)

    cl = asap_object.adata.load_datainfo_batch(1,0,asap_object.adata.uns['shape'][0])
    ct = getct(cl,sample)
    pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
    asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])

    
def analyze_nmf():

    asappy.generate_pseudobulk(asap_object,tree_depth=10)
    
    n_topics = 15 ## paper 
    asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
    
    # asap_adata = asappy.generate_model(asap_object,return_object=True)
    asappy.generate_model(asap_object)
    
    asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')
    
    cluster_resolution= 0.1 ## paper
    asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
    print(asap_adata.obs.cluster.value_counts())
    
    ## min distance 0.5 paper
    asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
    asappy.plot_umap(asap_adata,col='cluster',pt_size=0.1,ftype='png')

    cl = asap_adata.obs.index.values
    ct = getct(cl,sample)
    asap_adata.obs['celltype']  = pd.Categorical(ct)
    asappy.plot_umap(asap_adata,col='celltype',pt_size=0.1,ftype='png')
    
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
    asappy.plot_umap(asap_adata,col='celltypel2',pt_size=0.1,ftype='png')
 
    
    
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
    asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')

    #### cells by factor plot 

    pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
    df = pd.DataFrame(pmf2t['prop'])
    df.columns = ['t'+str(x) for x in df.columns]


    df.reset_index(inplace=True)
    cl = asap_adata.obs.index.values
    ct = ct = getct(cl,sample)
    df['celltype'] = ct

    def sample_n_rows(group):
        return group.sample(n=min(n, len(group)))

    n= 3000  ##24k in total
    sampled_df = df.groupby('celltype', group_keys=False).apply(sample_n_rows)
    sampled_df.reset_index(drop=True, inplace=True)
    print(sampled_df)

    sampled_df.drop(columns='index',inplace=True)
    sampled_df.set_index('celltype',inplace=True)

    import matplotlib.pylab as plt
    import seaborn as sns

    sns.clustermap(sampled_df,cmap='Oranges',col_cluster=False)
    plt.savefig(asap_adata.uns['inpath']+'_prop_hmap.png');plt.close()



#     ###################

    df = pd.DataFrame(asap_adata.obsm['corr'])
    df.columns = ['t'+str(x) for x in df.columns]
    clusts = [x for x in df.idxmax(axis=1)]

    from asappy.clustering.leiden import refine_clusts

    clusts = refine_clusts(df.to_numpy(),clusts,25)
    dfct = pd.DataFrame(clusts,columns=['topic'])
    dfct['celltype'] = ct
    dfct['score'] = 1

    ######### celltype proportion per topic
    dfct_grp = dfct.groupby(['topic','celltype']).count().reset_index()
    celltopic_sum = dict(dfct.groupby(['topic'])['score'].sum())
    dfct_grp['ncount'] = [x/celltopic_sum[y] for x,y in zip(dfct_grp['score'],dfct_grp['topic'])]


    custom_palette = get_colors(dfct_grp['celltype'].nunique())


    p = (ggplot(data=dfct_grp, mapping=aes(x='topic', y='ncount', fill='celltype')) +
        geom_bar(position="stack", stat="identity", size=0) +
        scale_fill_manual(values=custom_palette) )

    p = p + theme(
        plot_background=element_rect(fill='white'),
        panel_background = element_rect(fill='white'))
        # axis_text_x=element_blank())
    p.save(filename = asap_adata.uns['inpath']+'_topic_ctprop.pdf', height=5, width=15, units ='in', dpi=600)


