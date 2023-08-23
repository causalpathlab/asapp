import pandas as pd
import numpy as np
from plotnine import *
from ..util.analysis import get_topic_top_genes

def plot_umap(asap_adata,col='cluster'):
	
	df_umap = pd.DataFrame(asap_adata.obsm['umap_coords'],columns=['umap1','umap2'])
	df_umap[col] = pd.Categorical(asap_adata.obs[col].values)

	pt_size=0.1
	legend_size=7
	
	p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
		geom_point(size=pt_size) +
		scale_colour_brewer(type="qual", palette="Paired")+
		guides(color=guide_legend(override_aes={'size': legend_size})))
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
	p.save(filename = asap_adata.uns['inpath']+'_'+col+'_'+'umap.png', height=5, width=8, units ='in', dpi=300)

def plot_structure(asap_adata,mode):

	df = pd.DataFrame(asap_adata.obsm[mode])
	df.columns = ['t'+str(x) for x in df.columns]
	df.reset_index(inplace=True)
	df['cluster'] = asap_adata.obs['cluster'].values
	dfm = pd.melt(df,id_vars=['index','cluster'])
	dfm.columns = ['id','cluster','topic','value']

	dfm['id'] = pd.Categorical(dfm['id'])
	dfm['cluster'] = pd.Categorical(dfm['cluster'])
	dfm['topic'] = pd.Categorical(dfm['topic'])
	
	col_vector = [
    '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
    '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
    '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'
	]
	
	p = (ggplot(data=dfm, mapping=aes(x='id', y='value', fill='topic')) +
	    geom_bar(position="stack", stat="identity", size=0) +
	    scale_fill_manual(values=col_vector) +
    	facet_grid('~ cluster', scales='free', space='free'))
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
		axis_text_x=element_blank())
	p.save(filename = asap_adata.uns['inpath']+'_'+mode+'_'+'struct.png', height=5, width=15, units ='in', dpi=300)

def plot_gene_loading(asap_adata,max_thresh=None):

	import seaborn as sns
	import matplotlib.pylab as plt

	df_beta = pd.DataFrame(asap_adata.varm['beta'].T)
	df_beta.columns = asap_adata.var.index.values
	df_top = get_topic_top_genes(df_beta.iloc[:,:],top_n=3)
	df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
	if max_thresh:
		df_top[df_top>max_thresh] = max_thresh
	sns.clustermap(df_top.T,cmap='viridis')
	plt.savefig(asap_adata.uns['inpath']+'_beta.png');plt.close()

# def plot_genes_meanvar_barchart(dfm,batch_label,outfile,mode):

# 	df = dfm.copy()
# 	if mode == 'var':
# 		size = df.var(0).max()
# 		df['batch'] = batch_label
# 		df = df.groupby('batch').var(0)
# 	elif mode == 'mean':
# 		size = df.mean(0).max()
# 		df['batch'] = batch_label
# 		df = df.groupby('batch').mean(0)

# 	x = [size/y for y in [10000,1000,100,10,1]]
# 	y = []
# 	n = len(x)
# 	for indx,b in enumerate(df.index.values):

# 		data = df.loc[df.index==b,:].values[0]
# 		# data = np.sort(data) # optional in this case

# 		block_counts = []

# 		for i in range(n):
			
# 			if i ==0 :block_start = 0
# 			else: block_start = x[i-1]
			
# 			block_end = x[i]
# 			block = data[(data > block_start) & (data <= block_end)]
# 			block_counts.append(len(block))
# 		y.append(block_counts)
# 	dfp = pd.DataFrame(y).T
# 	dfp.columns = df.index.values
# 	dfp.index = x

# 	dfp = dfp.reset_index()
# 	dfpm = pd.melt(dfp,id_vars='index')
# 	print(dfp)
# 	print(dfpm)

# 	from plotnine import (
# 		ggplot,
# 		labs,
# 		aes,
# 		geom_bar,
# 		theme,
# 		element_rect
# 	)
# 	dfpm['index'] = [x[:6] for x in dfpm['index'].astype(str) ]
# 	p = ggplot(dfpm, aes(x='index', y='value', fill='variable')) + geom_bar(stat = "identity", position = "dodge")
# 	p = p + theme(
# 		plot_background=element_rect(fill='white'),
# 		panel_background = element_rect(fill='white')
# 	) + labs(title='Gene mean var analysis', x='Gene '+mode, y='Number of genes')
# 	p.save(filename = outfile, height=6, width=8, units ='in', dpi=300)

# def plot_gene_depthvar_barchart(dfm,batch_label,outfile,size=10):

# 	dfb = dfm.copy()
# 	dfb['batch'] = batch_label
# 	dfb = dfb.groupby('batch')

# 	df = dfm.copy()
# 	df['batch'] = batch_label
# 	df = df.groupby('batch').var(0)

# 	# x = [dfm.var(0).max()/y for y in [10000,1000,100,10,1]]
# 	x = [5/y for y in [10000,1000,100,10,1]]
# 	y = {}
# 	n = len(x)
# 	for indx,b in enumerate(df.index.values):

# 		data = df.loc[df.index==b,:].values[0]

# 		for i in range(n):
			
# 			if i ==0 : block_start = 0
# 			else: block_start = x[i-1]
			
# 			block_end = x[i]
# 			block = [(data > block_start) & (data <= block_end)]
	
# 			vn=5
# 			df_current = dfm.loc[dfb.groups[b],block[0]]
# 			total_var = df_current.var().sum() * (df_current.shape[0]-1)
# 			depth = df_current.sum(1).values
# 			sorted_depth = np.sort(depth)

# 			dblock_size = int(len(depth) / n)

# 			# Generate blocks and count occurrences
# 			dblocks = []

# 			for j in range(vn):
# 				dblock_start = j * dblock_size
# 				dblock_end = dblock_start + dblock_size
# 				dblock = [(depth >= sorted_depth[dblock_start]) & (depth < sorted_depth[dblock_end])]
# 				var_ratio =(df_current.loc[dblock[0]].var().sum() * (df_current.loc[dblock[0]].shape[0]-1)) / total_var
# 				dblocks.append(var_ratio)
# 			y[b+'_'+str(x[i])] = dblocks

# 	dfp = pd.DataFrame(y)
# 	dfp.index = ['group_'+str(x) for x in dfp.index.values]
# 	dfp = dfp.reset_index()
	
# 	dfpm = pd.melt(dfp,id_vars='index')
# 	print(dfp)

# 	dfpm['batch'] = [x.split('_')[0] for x in dfpm['variable']]

# 	dfpm['variable'] = [x.split('_')[1][:6] for x in dfpm['variable']]

# 	from plotnine import (
# 	ggplot,
# 	labs,
# 	aes,
# 	geom_bar,
# 	theme,
# 	element_rect,
# 	facet_wrap
# 	)
# 	p = ggplot(dfpm, aes(x='variable', y='value', fill='index')) + geom_bar(stat = "identity", position = "stack") + facet_wrap('~ batch', ncol=1,scales='free')
# 	p = p + theme(
# 		plot_background=element_rect(fill='white'),
# 		panel_background = element_rect(fill='white')
# 	) + labs(title='Gene depth var analysis', x='Cell groups ', y='variation')
# 	p.save(filename = outfile, height=6, width=8, units ='in', dpi=300)

def plot_stats(df,batch_label,outfile):

	from plotnine import (
		ggplot,
		ggtitle,
		aes,
		geom_density,
		theme,
		element_rect,
		facet_wrap
	)

	dfp = pd.DataFrame(df.sum(1),columns=['depth'])
	dfp['mean'] = df.mean(1)
	dfp['var'] = df.var(1)
	dfp['batch'] = batch_label

	dfpm = pd.melt(dfp,id_vars=['batch'])

	p = ggplot(dfpm, aes(x='value',fill='batch')) + geom_density(alpha=0.8) + facet_wrap('~ variable', ncol=1,scales='free')
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white')
	) + ggtitle('stats')
	p.save(filename = outfile, height=6, width=8, units ='in', dpi=300)

# def pbulk_hist(asap):
# 	if len(asap.pbulkd) ==1 :
# 		sns.histplot(x=[len(asap.pbulkd[x])for x in asap.pbulkd.keys()])
# 		plt.savefig(asap.adata.outpath+'_pbulk_hist_full.png')
# 		plt.close()
# 	else:
# 		pblen = []
# 		for k,v in enumerate(asap.pbulkd):
# 			for in_v in asap.pbulkd[v].values():
# 				pblen.append(len(in_v))
# 		sns.histplot(x=pblen)
# 		plt.savefig(asap.adata.outpath+'_pbulk_hist_batch.png')
# 		plt.close()

# def plot_pbulk_batchratio(df,outfile):

# 	from plotnine import (
# 		ggplot,
# 		aes,
# 		geom_bar,
# 		theme,
# 		theme_set,
# 		theme_void,
# 		element_rect
# 	)

# 	# theme_set(theme_void())
# 	df = df.reset_index().rename(columns={'index': 'pbindex'})
# 	dfm = pd.melt(df,id_vars='pbindex')
# 	dfm = dfm.sort_values(['variable','value'])

# 	p = ggplot(dfm, aes(x='pbindex', y='value',fill='variable')) + geom_bar(position="stack",stat="identity",size=0)
# 	p = p + theme(
# 		plot_background=element_rect(fill='white'),
# 		panel_background = element_rect(fill='white')
# 	)
# 	p.save(filename = outfile, height=3, width=8, units ='in', dpi=300)

# def plot_marker_genes(fn,mtx,rows,cols,df_umap,marker_genes):

# 	from anndata import AnnData
# 	import scanpy as sc
# 	import numpy as np

# 	import matplotlib.pylab as plt
# 	plt.rcParams['figure.figsize'] = [10, 5]
# 	plt.rcParams['figure.autolayout'] = True
# 	import seaborn as sns

# 	adata = AnnData(mtx)
# 	sc.pp.normalize_total(adata, target_sum=1e4)
# 	sc.pp.log1p(adata)
# 	dfn = adata.to_df()
# 	dfn.columns = cols
# 	dfn['cell'] = rows

# 	dfn = pd.merge(dfn,df_umap,on='cell',how='left')

# 	fig, ax = plt.subplots(2,3) 
# 	ax = ax.ravel()

# 	for i,g in enumerate(marker_genes):
# 		if g in dfn.columns:
# 			print(g)
# 			val = np.array([x if x<3 else 3.0 for x in dfn[g]])
# 			sns.scatterplot(data=dfn, x='umap1', y='umap2', hue=val,s=.1,palette="viridis",ax=ax[i],legend=False)

# 			norm = plt.Normalize(val.min(), val.max())
# 			sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
# 			sm.set_array([])

# 			# cax = fig.add_axes([ax[i].get_position().x1, ax[i].get_position().y0, 0.01, ax[i].get_position().height])
# 			fig.colorbar(sm,ax=ax[i])
# 			ax[i].axis('off')

# 			ax[i].set_title(g)
# 	fig.savefig(fn+'_umap_marker_genes_legend.png',dpi=600);plt.close()

