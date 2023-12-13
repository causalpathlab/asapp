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
from plotnine import * 
########################################

def calc_score(ct,cl):
	from sklearn.metrics import normalized_mutual_info_score
	from sklearn.metrics.cluster import adjusted_rand_score
	nmi =  normalized_mutual_info_score(ct,cl)
	ari = adjusted_rand_score(ct,cl)
	return nmi,ari


def getct(ids,sample,minor=False):
	ids = [x.replace('@'+sample,'') for x in ids]
	dfid = pd.DataFrame(ids,columns=['cell'])
	dfl = pd.read_csv(wdir+'results/'+sample+'_celltype.csv.gz')
	dfjoin = pd.merge(dfl,dfid,on='cell',how='right')
	ct = dfjoin['celltype'].values
	if minor:
		ct2 = dfjoin['celltype_minor'].values
	
	return ct,ct2

######################################################
import sys 
sample = 'brca'

wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_4_b/'

data_size = 110000
number_batches = 1


# asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)


def analyze_randomproject():
	
	from asappy.clustering.leiden import leiden_cluster
	from umap.umap_ import find_ab_params, simplicial_set_embedding


	### get random projection
	# rp_data,rp_index = asappy.generate_randomprojection(asap_object,tree_depth=10)
	# rp = rp_data[0]['1_0_25000']['rp_data']
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
	snn,cluster = leiden_cluster(df.to_numpy(),resolution=1.0)
	pd.Series(cluster).value_counts() 



	## umap cluster using neighbour graph
	min_dist = 0.6
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

	dfumap =  pd.read_csv(wdir+'results/bcra_rp_dfumap.csv.gz')
	
	legend_size = 7
	pt_size=0.5
	gradient_palette = get_colors(dfumap['celltype'].nunique())
	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette) 
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
  		legend_position='none')
 
	fname = asap_object.adata.uns['inpath']+'_rp_celltype_'+'umap_nolabel.png'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)

	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette)+ guides(color=guide_legend(override_aes={'size': legend_size})) 
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
 
	fname = asap_object.adata.uns['inpath']+'_rp_celltype_'+'umap_withlabel.pdf'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)



	dfumap.to_csv(wdir+'results/bcra_rp_dfumap.csv.gz',compression='gzip')

	asap_rp_s1,asap_rp_s2 =  calc_score([ str(x) for x in dfumap['celltype']],cluster)
	print('---RP-----')
	print('NMI:'+str(asap_rp_s1))
	print('ARI:'+str(asap_rp_s2))


def analyze_pseudobulk():
	asappy.generate_pseudobulk(asap_object,tree_depth=10)
	asappy.pbulk_cellcounthist(asap_object)

	cl = asap_object.adata.load_datainfo_batch(1,0,data_size)
	ct = getct(cl,sample)
	pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
	asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])

	df = pbulk_batchratio
	df.to_csv(wdir+'results/brca_pbulk_batchratio.csv.gz',compression='gzip',index=False)
	df = pd.read_csv(wdir+'results/brca_pbulk_batchratio.csv.gz')
	### filter celltype with <1000 cells 
	df = df.reset_index().rename(columns={'index': 'pbindex'})
	dfm = pd.melt(df,id_vars='pbindex')
	dfm = dfm.sort_values(['variable'])

	gradient_palette = get_colors(dfm['variable'].nunique())	
 
	p = (ggplot(dfm, aes(x='pbindex', y='value',fill='variable')) + 
		geom_bar(position="stack",stat="identity",size=0)+
  		scale_fill_manual(values=gradient_palette) 
  )
 
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),legend_position='none')
 
	p.save(filename = asap_object.adata.uns['inpath']+'_pbulk_ratio.png', height=4, width=8, units ='in', dpi=600)
	# p.save(filename = asap_object.adata.uns['inpath']+'_pbulk_ratio.pdf', height=4, width=8, units ='in', dpi=600)
	
	  
def analyze_nmf():

	asappy.generate_pseudobulk(asap_object,tree_depth=10)
	
	n_topics = 25 ## paper 
	asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
	
	# asap_adata = asappy.generate_model(asap_object,return_object=True)
	asappy.generate_model(asap_object)
	
	asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')
	
	cluster_resolution= 0.1 ## paper
	asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
	print(asap_adata.obs.cluster.value_counts())
	
	## min distance 0.5 paper
	asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
	asappy.plot_umap(asap_adata,col='cluster',pt_size=0.5,ftype='png')

	cl = asap_adata.obs.index.values
	ct, ct2 = getct(cl,sample,minor=True)
	asap_adata.obs['celltype']  = pd.Categorical(ct)
	asap_adata.obs['celltype_m']  = pd.Categorical(ct2)
	asappy.plot_umap(asap_adata,col='celltype',pt_size=0.5,ftype='png')
	asappy.plot_umap(asap_adata,col='celltype_m',pt_size=0.5,ftype='png')
	
	asap_adata.write(wdir+'results/'+sample+'.h5asapad')
	
	## top 10 main paper
	asappy.plot_gene_loading(asap_adata,top_n=10,max_thresh=25)
	
	asap_s1,asap_s2 =  calc_score([str(x) for x in asap_adata.obs['celltype'].values],asap_adata.obs['cluster'].values)

	print('---NMF-----')
	print('NMI:'+str(asap_s1))
	print('ARI:'+str(asap_s2))

 	dfumap = pd.DataFrame(asap_adata.obsm['umap_coords'])
	dfumap['cell'] = asap_adata.obs.index.values
	dfumap.columns = ['umap1','umap2','cell']
	dfumap['celltype'] = asap_adata.obs['celltype'].values
	
	
 	legend_size = 7
	pt_size=0.5
	gradient_palette = get_colors(dfumap['celltype'].nunique())
	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette) 
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
  		legend_position='none')
 
	fname = asap_object.adata.uns['inpath']+'_nmf_'+'umap_nolabel.png'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)

	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette)+ guides(color=guide_legend(override_aes={'size': legend_size})) 
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
 
	fname = asap_object.adata.uns['inpath']+'_nmf_'+'umap_withlabel.pdf'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)

	dfumap['celltype'] = asap_adata.obs['celltype_m'].values
	
	
 	legend_size = 7
	pt_size=0.5
	gradient_palette = get_colors(dfumap['celltype'].nunique())
	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette) 
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
  		legend_position='none')
 
	fname = asap_object.adata.uns['inpath']+'_nmf2_'+'umap_nolabel.png'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)

	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette)+ guides(color=guide_legend(override_aes={'size': legend_size})) 
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
 
	fname = asap_object.adata.uns['inpath']+'_nmf2_'+'umap_withlabel.pdf'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)

analyze_randomproject()

analyze_pseudobulk()

analyze_nmf()

def extra(): 
	asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')

	#### cells by factor plot 

	pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
	df = pd.DataFrame(pmf2t['prop'])
	df.columns = ['t'+str(x) for x in df.columns]


	df.reset_index(inplace=True)
	cl = asap_adata.obs.index.values
	ct,ct2 = getct(cl,sample,minor=True)
	df['celltype'] = ct

	def sample_n_rows(group):
		return group.sample(n=min(n, len(group)))

	n= 3000 ## per cell type 
	### paper - take all cells for theta
	
	sampled_df = df.groupby('celltype', group_keys=False).apply(sample_n_rows)
	sampled_df.reset_index(drop=True, inplace=True)
	print(sampled_df)

	sampled_df.drop(columns='index',inplace=True)
	sampled_df.set_index('celltype',inplace=True)

	import matplotlib.pylab as plt
	import seaborn as sns

	sns.clustermap(sampled_df,cmap='Oranges',col_cluster=False)
	plt.savefig(asap_adata.uns['inpath']+'_prop_hmap.png',dpi=600);plt.close()



	###################
	# n= 3125
	# sampled_df = df.groupby('celltype', group_keys=False).apply(sample_n_rows)
	# sampled_df.reset_index(drop=True, inplace=True)
	# print(sampled_df)


	# dfm = pd.melt(sampled_df,id_vars=['index','celltype'])
	# dfm.columns = ['id','celltype','topic','value']

	# dfm['id'] = pd.Categorical(dfm['id'])
	# dfm['celltype'] = pd.Categorical(dfm['celltype'])
	# dfm['topic'] = pd.Categorical(dfm['topic'])

	# custom_palette = [
	# "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
	# "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
	# "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
	# "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
	# '#6b6ecf', '#8ca252',  '#8c6d31', '#bd9e39', '#d6616b'
	# ]

	# p = (ggplot(data=dfm, mapping=aes(x='id', y='value', fill='topic')) +
	#     geom_bar(position="stack", stat="identity", size=0) +
	#     scale_fill_manual(values=custom_palette) +
	#     facet_grid('~ celltype', scales='free', space='free'))

	# p = p + theme(
	#     plot_background=element_rect(fill='white'),
	#     panel_background = element_rect(fill='white'),
	#     axis_text_x=element_blank())
	# p.save(filename = asap_adata.uns['inpath']+'_sc_topic_struct.png', height=5, width=15, units ='in', dpi=300)

	###################
	pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
	df = pd.DataFrame(asap_adata.obsm['corr'])
	df.columns = ['t'+str(x) for x in df.columns]
	clusts = [x for x in df.idxmax(axis=1)]

	from asappy.clustering.leiden import refine_clusts

	clusts = refine_clusts(df.to_numpy(),clusts,25)
	
	dfct = pd.DataFrame(clusts,columns=['topic'])
	dfct['celltype'] = ct
	dfct['score'] = 1

	######### celltype proportion per topic
	
	custom_palette = [
	"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
	"#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
	"#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
	"#c49c94", "#f7b6d2", "#c7c7c7", "#9edae5","#dbdb8d", 
	'#6b6ecf', '#8ca252',  '#8c6d31', '#bd9e39', '#d6616b'
	]
		
	dfct_grp = dfct.groupby(['topic','celltype']).count().reset_index()
	celltopic_sum = dict(dfct.groupby(['topic'])['score'].sum())
	dfct_grp['ncount'] = [x/celltopic_sum[y] for x,y in zip(dfct_grp['score'],dfct_grp['topic'])]

	dfct_grp = dfct_grp.sort_values(['score','celltype'],ascending=False)
	torder = dfct_grp['topic'].unique()


	p = (ggplot(data=dfct_grp, mapping=aes(x='topic', y='ncount', fill='celltype')) +
		geom_bar(position="stack", stat="identity", size=0) +
		scale_fill_manual(values=custom_palette) )
	
	p = p + scale_x_discrete(limits=torder)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
		# axis_text_x=element_blank())
	p.save(filename = asap_adata.uns['inpath']+'_topic_ctprop.pdf', height=5, width=15, units ='in', dpi=600)

	p = (ggplot(data=dfct_grp, mapping=aes(x='topic', y='score', fill='celltype')) +
		geom_bar(position="stack", stat="identity", size=0) +
		scale_fill_manual(values=custom_palette) )
	p = p + scale_x_discrete(limits=torder)
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
		# axis_text_x=element_blank())
	p.save(filename = asap_adata.uns['inpath']+'_topic_ctprop_count.pdf', height=5, width=15, units ='in', dpi=600)




	############# topic propertion per cell type
	
	dfct = pd.DataFrame(clusts,columns=['topic'])
	dfct['celltype'] = ct
	dfct['score'] = 1

	dfct_grp = dfct.groupby(['celltype','topic']).count().reset_index()


	celltopic_sum = dict(dfct.groupby(['celltype'])['score'].sum())
	dfct_grp['ncount'] = [x/celltopic_sum[y] for x,y in zip(dfct_grp['score'],dfct_grp['celltype'])]



	dfct_grp = dfct_grp.sort_values(['score','topic'],ascending=False)
	torder = dfct_grp['celltype'].unique()


	p = (ggplot(data=dfct_grp, mapping=aes(x='celltype', y='ncount', fill='topic')) +
		geom_bar(position="stack", stat="identity", size=0) +
		scale_fill_manual(values=custom_palette) )
	
	p = p + scale_x_discrete(limits=torder)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
		# axis_text_x=element_blank())
	p.save(filename = asap_adata.uns['inpath']+'_ct_topicprop.pdf', height=5, width=15, units ='in', dpi=600)

	p = (ggplot(data=dfct_grp, mapping=aes(x='celltype', y='score', fill='topic')) +
		geom_bar(position="stack", stat="identity", size=0) +
		scale_fill_manual(values=custom_palette) )
	p = p + scale_x_discrete(limits=torder)
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
		# axis_text_x=element_blank())
	p.save(filename = asap_adata.uns['inpath']+'_ct_topicprop_count.pdf', height=5, width=15, units ='in', dpi=600)


