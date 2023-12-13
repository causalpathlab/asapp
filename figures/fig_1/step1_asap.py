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
	dfl = pd.read_csv(wdir+'results/'+sample+'_celltype.csv.gz')
	dfl['cell'] = [x.replace('@'+sample,'') for x in dfl['cell']]
	dfjoin = pd.merge(dfl,dfid,on='cell',how='right')
	ct = dfjoin['celltype'].values
	
	return ct

######################################################
import sys 
sample = 'pancreas'

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_1/'

data_size = 20000
number_batches = 1


asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)


def analyze_randomproject():
	
	from asappy.clustering.leiden import kmeans_cluster
	from umap.umap_ import find_ab_params, simplicial_set_embedding
	from umap import UMAP


	## get random projection
	rp_data = asappy.generate_randomprojection(asap_object,tree_depth=10)
	rp_data = rp_data['full']
	rpi = asap_object.adata.load_datainfo_batch(1,0,rp_data.shape[0])
	df=pd.DataFrame(rp_data)
	df.index = rpi


	### draw nearest neighbour graph and cluster
	
	
	cluster = kmeans_cluster(df.to_numpy(),k=10)
	pd.Series(cluster).value_counts() 
 	
	umap_2d = UMAP(n_components=2, init='random', random_state=0,min_dist=1.0)

	proj_2d = umap_2d.fit_transform(df.to_numpy())


	dfumap = pd.DataFrame(proj_2d[:,])
	dfumap['cell'] = rpi
	dfumap.columns = ['umap1','umap2','cell']

	ct = getct(dfumap['cell'].values,sample)
	dfumap['celltype'] = pd.Categorical(ct)
	dfumap['cluster'] = cluster
 
	dfumap = pd.read_csv(wdir+'results/_dfumap_rp.csv.gz')
	pt_size = 2.5
 	legend_size = 7
	gradient_palette =  ['#F3E33B',
	'#e9a123',
	'#f47e17',
	'#d85a3b',
	'#908355',
	'#4d4376',
	'#ba3339',
	'#ded3b6',
	'#db739b',
	'#7b362d']
 
	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette)+ 
		guides(color=guide_legend(override_aes={'size': legend_size}))
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white')
  		)
 
	fname = asap_object.adata.uns['inpath']+'_rp_celltype_'+'umap.pdf'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)

	dfumap.to_csv(wdir+'results/_dfumap_rp.csv.gz',compression='gzip')


def analyze_pseudobulk():
	asappy.generate_pseudobulk(asap_object,tree_depth=10)
	asappy.pbulk_cellcounthist(asap_object)

	cl = asap_object.adata.load_datainfo_batch(1,0,asap_object.adata.uns['shape'][0])
	ct = getct(cl,sample)
	pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
	# theme_set(theme_void())
	df = pbulk_batchratio
	df = df.reset_index().rename(columns={'index': 'pbindex'})
	dfm = pd.melt(df,id_vars='pbindex')
	dfm = dfm.sort_values(['variable','value'])

	p = (ggplot(dfm, aes(x='pbindex', y='value',fill='variable')) + 
		geom_bar(position="stack",stat="identity",size=0) +
		scale_fill_manual(values=gradient_palette) 		)
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white')
	)
	p.save(filename = 'results/_pbulk_ratio.png', height=4, width=8, units ='in', dpi=600)
	
	
def analyze_nmf():

	asappy.generate_pseudobulk(asap_object,tree_depth=10)
	
	n_topics = 12 ## paper 
	asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
	
	# asap_adata = asappy.generate_model(asap_object,return_object=True)
	asappy.generate_model(asap_object)
	
	asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')
	
    cluster_resolution= 0.3 ## paper
    asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
    print(asap_adata.obs.cluster.value_counts())
    
    ## min distance 0.5 paper
    asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.3)
	cl = asap_adata.obs.index.values
    ct = getct(cl,sample)
    asap_adata.obs['celltype']  = pd.Categorical(ct)
	
 	col='celltype'
	df_umap = pd.DataFrame(asap_adata.obsm['umap_coords'],columns=['umap1','umap2'])
	df_umap[col] = pd.Categorical(asap_adata.obs[col].values)
	nlabel = asap_adata.obs[col].nunique()

 
	fname = asap_adata.uns['inpath']+'_'+col+'_'+'umap.pdf'

	legend_size=7

	p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette)  +
		guides(color=guide_legend(override_aes={'size': legend_size})))
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
	

	p.save(filename = fname, height=8, width=15, units ='in', dpi=600)

 	col='cluster'
	df_umap = pd.DataFrame(asap_adata.obsm['umap_coords'],columns=['umap1','umap2'])
	df_umap[col] = pd.Categorical(asap_adata.obs[col].values)
	nlabel = asap_adata.obs[col].nunique()

 
	fname = asap_adata.uns['inpath']+'_'+col+'_'+'umap.pdf'

	legend_size=7

	p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette)  +
		guides(color=guide_legend(override_aes={'size': legend_size})))
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
	

	p.save(filename = fname, height=8, width=15, units ='in', dpi=600)





    
    asap_adata.write(wdir+'results/'+sample+'.h5asapad')

# analyze_randomproject()

# analyze_pseudobulk()

# analyze_nmf()

    asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')

	from asappy.util.analysis import get_topic_top_genes,row_col_order
    import matplotlib.pylab as plt
    import seaborn as sns

 
	top_n=10
 	max_thresh=15
 	df_beta = pd.DataFrame(asap_adata.varm['beta'].T)
	df_beta.columns = asap_adata.var.index.values
	df_beta = df_beta.loc[:, ~df_beta.columns.duplicated(keep='first')]
	df_top = get_topic_top_genes(df_beta.iloc[:,:],top_n)
	df_beta = df_beta.loc[:,df_top['Gene'].unique()]
	ro,co = row_col_order(df_beta)
	df_beta = df_beta.loc[ro,co]
	df_beta[df_beta>max_thresh] = max_thresh
	sns.clustermap(df_beta.T,cmap='Reds')
	plt.savefig(asap_adata.uns['inpath']+'_beta'+'_th_'+str(max_thresh)+'.png');plt.close()
    #### cells by factor plot 

    pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.uns['pseudobulk']['pb_theta'])
    df = pd.DataFrame(pmf2t['prop'])
    df.columns = ['t'+str(x) for x in df.columns]




	from matplotlib.colors import ListedColormap

	# Define the custom colormap
	cmap = ListedColormap(sns.color_palette(['white', 'yellow'], n_colors=256))


    sns.clustermap(df.T,cmap='Oranges')
    plt.savefig(asap_adata.uns['inpath']+'_bulk_theta_prop_hmap.png');plt.close()




    asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')

    #### cells by factor plot 

    pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
    df = pd.DataFrame(pmf2t['prop'])
    df.columns = ['t'+str(x) for x in df.columns]


    ct =  getct(cl,sample)
    df['celltype'] = ct
    df.set_index('celltype',inplace=True)


    sns.clustermap(df.T,cmap='Oranges')
    plt.savefig(asap_adata.uns['inpath']+'_prop_hmap.png');plt.close()


