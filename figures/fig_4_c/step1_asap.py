from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import asappy
import pandas as pd
import numpy as np
import anndata as an
from umap import UMAP
from plotnine import *
from asappy.plotting.palette import get_colors
########################################

curate_ct ={
'fibroblast':'fibroblast',
'T cell':'T cell',
'mature NK T cell':'T cell',
'mesenchymal stem cell':'stem cell',
'macrophage':'macrophage',
'plasma cell':'plasma cell',
'endothelial cell':'endothelial cell',
'myofibroblast cell':'myofibroblast cell',
'neutrophil':'neutrophil',
'leukocyte':'leukocyte',
'mast cell':'mast cell',
'smooth muscle cell':'muscle cell',
'B cell':'B cell',
'keratinocyte':'keratinocyte',
'epithelial cell':'epithelial cell',
'basal cell':'basal cell',
'vein endothelial cell':'endothelial cell',
'pericyte':'pericyte',
'capillary endothelial cell':'endothelial cell',
'endothelial cell of artery':'endothelial cell',
'endothelial cell of lymphatic vessel':'endothelial cell',
'erythrocyte':'erythrocyte',
'bladder urothelial cell':'urothelial cell',
'monocyte':'monocyte',
'naive thymus-derived CD4-positive, alpha-beta T cell':'CD4 T cell',
'CD4-positive, alpha-beta T cell':'CD4 T cell',
'CD8-positive, alpha-beta cytokine secreting effector T cell':'CD8 T cell',
'classical monocyte':'monocyte',
'CD4-positive, alpha-beta memory T cell':'CD4 T cell',
'CD8-positive, alpha-beta T cell':'CD8 T cell',
'naive B cell':'B cell',
'type I NK T cell':'NK cell',
'memory B cell':'B cell',
'non-classical monocyte':'monocyte',
'cardiac muscle cell':'muscle cell',
'native cell':'stem cell',
'cardiac endothelial cell':'endothelial cell',
'conjunctival epithelial cell':'epithelial cell',
'corneal epithelial cell':'epithelial cell',
'stromal cell':'stromal cell',
'pancreatic acinar cell':'acinar cell',
'pancreatic ductal cell':'ductal cell',
'myeloid cell':'myeloid cell',
'effector CD8-positive, alpha-beta T cell':'CD8 T cell',
'effector CD4-positive, alpha-beta T cell':'CD4 T cell',
'innate lymphoid cell':'lymphoid cell',
'CD8-positive, alpha-beta memory T cell':'CD8 T cell',
'regulatory T cell':'T cell',
'kidney epithelial cell':'epithelial cell',
'CD4-positive helper T cell':'CD4 T cell',
'enterocyte of epithelium of large intestine':'epithelial cell',
'enterocyte':'enterocyte',
'hepatocyte':'hepatocyte',
'club cell':'club cell',
'type II pneumocyte':'pneumocyte',
'vascular associated smooth muscle cell':'muscle cell',
'luminal epithelial cell of mammary gland':'epithelial cell',
'fibroblast of breast':'fibroblast',
'skeletal muscle satellite stem cell':'stem cell',
'endothelial cell of vascular tree':'endothelial cell',
'acinar cell of salivary gland':'acinar cell',
'duct epithelial cell':'epithelial cell',
'luminal cell of prostate epithelium':'epithelial cell',
'basal cell of prostate epithelium':'epithelial cell',
'enterocyte of epithelium of small intestine':'epithelial cell',
'DN3 thymocyte':'DN3 thymocyte',
}


def calc_score(ct,cl):
	from sklearn.metrics import normalized_mutual_info_score
	from sklearn.metrics.cluster import adjusted_rand_score
	nmi =  normalized_mutual_info_score(ct,cl)
	ari = adjusted_rand_score(ct,cl)
	return nmi,ari


def getct(ids,sample):
# '''
	import h5py as hf
	f = hf.File('/data/sishir/data/tabula_sapiens/tabsap_all.h5ad','r')
# 	cell_ids = [x.decode('utf-8') for x in f['obs']['_index']]
# 	codes = list(f['obs']['cell_type']['codes'])
# 	cat = [x.decode('utf-8') for x in f['obs']['cell_type']['categories']]
# 	f.close()

# 	catd ={}
# 	for ind,itm in enumerate(cat):catd[ind]=itm

# 	cell_type = [ [x,catd[y]] for x,y in zip(cell_ids,codes)]

# 	df_ct = pd.DataFrame(cell_type,columns=['cell','celltype'])
# 	df_ct.to_csv('/data/sishir/data/tabula_sapiens/tabsap_celltype.csv.gz',index=False,compression='gzip')
	
# '''
	ids = [x.replace('@'+sample,'') for x in ids]
	dfid = pd.DataFrame(ids,columns=['cell'])
	dfl = pd.read_csv(wdir+'results/'+sample+'_celltype.csv.gz')
	dfjoin = pd.merge(dfl,dfid,on='cell',how='right')
	ct = dfjoin['celltype'].values    
	return ct

######################################################
import sys 
sample = 'tabsap'

wdir = 'experiments/asapp/figures/fig_4_c/'


## for rp data
data_size = 50100 
number_batches = 10
# data_size = 100100 
# number_batches = 5



# asappy.create_asap_data(sample,working_dirpath=wdir)

asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)


def analyze_randomproject():
	
	from asappy.clustering.leiden import kmeans_cluster
	from umap import UMAP

	### get random projection
	rp_data,rp_index = asappy.generate_randomprojection(asap_object,tree_depth=10)
	rp = rp_data[0]['1_0_100100']['rp_data']
	for d in rp_data[1:]:
		rp = np.vstack((rp,d[list(d.keys())[0]]['rp_data']))
	rpi = np.array(rp_index[0])
	for i in rp_index[1:]:rpi = np.hstack((rpi,i))
	df=pd.DataFrame(rp)
	df.index = rpi

	cluster = kmeans_cluster(df.to_numpy(),k=161)
	pd.Series(cluster).value_counts() 
 
	df.to_csv(wdir+'results/tabsap_rp.csv.gz',compression='gzip')
	pd.DataFrame(cluster,columns=['cluster']).to_csv(wdir+'results/tabsap_rp_kmeanscluster.csv.gz',compression='gzip')

	
	umap_2d = UMAP(n_components=2, init='random', random_state=0,min_dist=0.1)

	proj_2d = umap_2d.fit_transform(df.to_numpy())


	dfumap = pd.DataFrame(proj_2d[:,])
	dfumap['cell'] = rpi
	dfumap.columns = ['umap1','umap2','cell']

	ct = getct(dfumap['cell'].values,sample)
	dfumap['celltype'] = pd.Categorical(ct)
	dfumap['cluster'] = cluster
 
	dfumap = pd.read_csv(wdir+'results/tabsap_rp_dfumap.csv.gz')
 
  	dftemp = dfumap.celltype.value_counts()  
   
	selected_celltype = dftemp[dftemp>1000].index.values
	
	dfumap = dfumap[dfumap['celltype'].isin(selected_celltype)]

 	dfumap['celltype'] = [curate_ct[x] for x in dfumap['celltype']]
	dfumap = dfumap.sort_values('celltype')
	
	pt_size = 0.05

	gradient_palette = get_colors(dfumap['celltype'].nunique())	
	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette) 
		# guides(color=guide_legend(override_aes={'size': legend_size}))
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
  		legend_position='none')
 
	fname = asap_object.adata.uns['inpath']+'_rp_celltype_'+'umap.png'
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
	
	print(asap_object.adata.uns['shape'])
	asappy.generate_pseudobulk(asap_object,tree_depth=10)
	asappy.pbulk_cellcounthist(asap_object)

	cl = asap_object.adata.load_datainfo_batch(1,0,asap_object.adata.uns['shape'][0])
	ct = getct(cl,sample)
 
	pbulk_batchratio  = asappy.get_psuedobulk_batchratio(asap_object,ct)
	# asappy.plot_pbulk_celltyperatio(pbulk_batchratio,asap_object.adata.uns['inpath'])
	df = pbulk_batchratio
	# df.to_csv(wdir+'results/tabsap_pbulk_batchratio.csv.gz',compression='gzip')
	df = pd.read_csv(wdir+'results/tabsap_pbulk_batchratio.csv.gz')
	### filter celltype with <1000 cells 
	df = df[selected_celltype]
	df = df.reset_index().rename(columns={'index': 'pbindex'})
	dfm = pd.melt(df,id_vars='pbindex')
	dfm['variable'] = [curate_ct[x] for x in dfm['variable']]
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

	n_topics = 100 ## paper 
	asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
	
	asappy.generate_model(asap_object)
	
	asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')
	
	mtx = asap_adata.obsm['corr']
	mtx = (mtx - np.mean(mtx, axis=0)) / np.std(mtx, axis=0, ddof=1)
	cluster = kmeans_cluster(mtx,k=161)
	pd.Series(cluster).value_counts() 

	umap_2d = UMAP(n_components=2, init='random', random_state=0,min_dist=0.1)
	proj_2d = umap_2d.fit_transform(mtx)


	dfumap = pd.DataFrame(proj_2d[:,])
	dfumap['cell'] = asap_adata.obs.index.values
	dfumap.columns = ['umap1','umap2','cell']
	ct = getct(dfumap['cell'].values,sample)
	dfumap['celltype'] = ct
	dfumap['cluster'] = cluster

	import h5py as hf
	f = hf.File('/data/sishir/data/tabula_sapiens/tabsap_all.h5ad','r')
	cell_ids = [x.decode('utf-8') for x in f['obs']['_index']]
	codes = list(f['obs']['tissue']['codes'])
	cat = [x.decode('utf-8') for x in f['obs']['tissue']['categories']]
	f.close()

	catd ={}
	for ind,itm in enumerate(cat):catd[ind]=itm

	tissue_type = [ [x,catd[y]] for x,y in zip(cell_ids,codes)]

	df_tisue = pd.DataFrame(tissue_type,columns=['cell','tissuetype'])

	ids = [x.replace('@'+sample,'') for x in dfumap['cell']]
	dfid = pd.DataFrame(ids,columns=['cell'])

	dfjoin = pd.merge(df_tisue,dfid,on='cell',how='right')
	tissue = dfjoin['tissuetype'].values

	dfumap['tissue_type'] = tissue 

	
	dfumap.to_csv(wdir+'results/tabsap_nmf_dfumap.csv.gz',compression='gzip')
 
	dfumap = pd.read_csv(wdir+'results/tabsap_nmf_dfumap.csv.gz')

 	dftemp = dfumap.celltype.value_counts()  
	selected_celltype = dftemp[dftemp>1000].index.values
	
	dfumap = dfumap[dfumap['celltype'].isin(selected_celltype)]


	# unq_ct = dfumap['celltype'].unique()
	# for x in unq_ct:
	# 	print("'"+x+':'+"'"+x+"'") 

	dfumap['celltype'] = [curate_ct[x] for x in dfumap['celltype']]
 
	pt_size = 0.05

	dfumap = dfumap.sort_values('celltype')
 
 
 	gradient_palette = get_colors(dfumap['celltype'].nunique())	

	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette) 
		# guides(color=guide_legend(override_aes={'size': legend_size}))
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
  		legend_position='none')
 
	fname = asap_adata.uns['inpath']+'_nmf_'+'umap.png'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)

	legend_size = 7
	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='celltype')) +
		geom_point(size=pt_size) +
		scale_color_manual(values=gradient_palette) +
		guides(color=guide_legend(override_aes={'size': legend_size}))
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
 
	fname = asap_adata.uns['inpath']+'_nmf_withlabel_'+'umap.pdf'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600,limitsize=False)

	# dfumap['cluster'] = pd.Categorical(dfumap['cluster'])
	# legend_size=7
	# p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='cluster')) +
	# 	geom_point(size=pt_size)+ 
	# 	scale_color_manual(values=gradient_palette) +
	# 	guides(color=guide_legend(override_aes={'size': legend_size}))
	# )
	
	# p = p + theme(
	# 	plot_background=element_rect(fill='white'),
	# 	panel_background = element_rect(fill='white'))
 
	# fname = asap_adata.uns['inpath']+'_nmf_cluster_'+'umap.png'
	# p.save(filename = fname, height=10, width=15, units ='in', dpi=600)
 

	dfumap = dfumap.sort_values('tissue_type')
	legend_size=7
	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='tissue_type')) +
		geom_point(size=pt_size)+ 
		scale_color_manual(values=gradient_palette) 
		
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
  		legend_position='none')
 
	fname = asap_adata.uns['inpath']+'_nmf_tissue_'+'umap.png'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)
 
	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color='tissue_type')) +
		geom_point(size=pt_size)+ 
		scale_color_manual(values=gradient_palette) +
		guides(color=guide_legend(override_aes={'size': legend_size}))
	)
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
 
	fname = asap_adata.uns['inpath']+'_nmf_tissue_'+'umap.pdf'
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)
 

	# ## top 10 main paper
	## total genes 
	asappy.plot_gene_loading(asap_adata,top_n=10,max_thresh=25)
	


	# asap_s1,asap_s2 =  calc_score([str(x) for x in asap_adata.obs['celltype'].values],asap_adata.obs['cluster'].values)

	# print('---NMF-----')
	# print('NMI:'+str(asap_s1))
	# print('ARI:'+str(asap_s2))

# analyze_randomproject()

analyze_pseudobulk()

# analyze_nmf()


def extra(): 
	asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')
	dfumap = pd.read_csv(wdir+'results/tabsap_nmf_dfumap.csv.gz')
	#### cells by factor plot 

	pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
	df = pd.DataFrame(pmf2t['prop'])
	df.columns = ['t'+str(x) for x in df.columns]


	df.reset_index(inplace=True)
	cl = asap_adata.obs.index.values
	ct = getct(cl,sample)
	df['celltype'] = ct
 
 	dftemp = df.celltype.value_counts()  
	selected_celltype = dftemp[dftemp>1000].index.values

 	df = df[df['celltype'].isin(selected_celltype)]
	df['celltype'] = [curate_ct[x] for x in df['celltype']]
 
	
	dfumap = dfumap[dfumap['celltype'].isin(selected_celltype)]
 
	dfumap['celltype'] = [curate_ct[x] for x in dfumap['celltype']]
	
	def sample_n_rows(group):
		return group.sample(n=min(n, len(group)))

	#### theta paper heatmap
	n= 1000 ## 3k per cell type for 32 celltypes with total cell >1k
	### total cells 483152 and after keeping 66 celltypes total cells 458521 with loss of just 24631 cells from low numberred cell type
	## sample is 32000 cells
 
	sampled_df = dfumap.groupby('celltype', group_keys=False).apply(sample_n_rows)
	sampled_df = df[df['index'].isin(sampled_df['Unnamed: 0'].values)]
	sampled_df.reset_index(drop=True, inplace=True)
	print(sampled_df)

	sampled_df.drop(columns='index',inplace=True)
	sampled_df.set_index('celltype',inplace=True)

	import matplotlib.pylab as plt
	import seaborn as sns

	sns.clustermap(sampled_df,cmap='Oranges',col_cluster=False)
	plt.savefig(asap_adata.uns['inpath']+'_prop_hmap.png',dpi=600);plt.close()

