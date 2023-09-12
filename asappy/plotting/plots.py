import pandas as pd
import numpy as np
from plotnine import *
import seaborn as sns
import matplotlib.pylab as plt

from ..util.analysis import get_topic_top_genes

custom_palette = [
"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
"#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
"#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
"#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
'#6b6ecf', '#8ca252',  '#8c6d31', '#bd9e39', '#d6616b'
]

custom_palette50 = [
"#556b2f","#a0522d","#228b22","#7f0000","#191970",
"#808000","#3cb371","#008080","#b8860b","#4682b4",
"#d2691e","#9acd32","#00008b","#32cd32","#7f007f",
"#8fbc8f","#b03060","#d2b48c","#9932cc","#ff4500",
"#ff8c00","#ffd700","#6a5acd","#ffff00","#0000cd",
"#00ff00","#00fa9a","#dc143c","#00ffff","#00bfff",
"#f4a460","#a020f0","#f08080","#adff2f","#ff6347",
"#da70d6","#ff00ff","#1e90ff","#f0e68c","#dda0dd",
"#90ee90","#87ceeb","#ff1493","#7fffd4","#ff69b4",
"#ffc0cb","#000000","#808080","#dcdcdc","#2f4f4f"
]

def plot_umap(asap_adata,col):
	
	df_umap = pd.DataFrame(asap_adata.obsm['umap_coords'],columns=['umap1','umap2'])
	df_umap[col] = pd.Categorical(asap_adata.obs[col].values)
	nlabel = asap_adata.obs[col].nunique() 
	fname = asap_adata.uns['inpath']+'_'+col+'_'+'umap.png'

	pt_size=1.0
	legend_size=7

	
	if nlabel <= 25 :

		cp = custom_palette[:nlabel]
		
		
		p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
			geom_point(size=pt_size) +
			scale_color_manual(values=custom_palette)  +
			guides(color=guide_legend(override_aes={'size': legend_size})))
		
		p = p + theme(
			plot_background=element_rect(fill='white'),
			panel_background = element_rect(fill='white'))
		

	else :
		p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
			geom_point(size=pt_size) +
			scale_color_manual(values=custom_palette50) +
			guides(color=guide_legend(override_aes={'size': legend_size})))
		
		p = p + theme(
			plot_background=element_rect(fill='white'),
			panel_background = element_rect(fill='white'))
		
	p.save(filename = fname, height=8, width=15, units ='in', dpi=300)


def plot_umap_df(df_umap,col,fpath):
	
	nlabel = df_umap[col].nunique() 
	fname = fpath+'_'+col+'_'+'umap.png'

	pt_size=1.0
	legend_size=7

	
	if nlabel <= 25 :

		cp = custom_palette[:nlabel]
		
		
		p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
			geom_point(size=pt_size) +
			scale_color_manual(values=custom_palette)  +
			guides(color=guide_legend(override_aes={'size': legend_size})))
		
		p = p + theme(
			plot_background=element_rect(fill='white'),
			panel_background = element_rect(fill='white'))
		

	else :
		p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
			geom_point(size=pt_size) +
			scale_color_manual(values=custom_palette50)+
			guides(color=guide_legend(override_aes={'size': legend_size})))
		
		p = p + theme(
			plot_background=element_rect(fill='white'),
			panel_background = element_rect(fill='white'))
		
	p.save(filename = fname, height=10, width=15, units ='in', dpi=300)

def plot_randomproj(dfrp,col,fname):
	sns.set(style="ticks")
	sns.pairplot(dfrp, kind='scatter',hue=col,palette=sns.color_palette("Paired"),plot_kws = {"s":5})
	plt.savefig(fname+'_rproj.png');plt.close()

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
	
	
	p = (ggplot(data=dfm, mapping=aes(x='id', y='value', fill='topic')) +
		geom_bar(position="stack", stat="identity", size=0) +
		scale_color_manual(values=custom_palette) +
		facet_grid('~ cluster', scales='free', space='free'))
	
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'),
		axis_text_x=element_blank())
	p.save(filename = asap_adata.uns['inpath']+'_'+mode+'_'+'struct.png', height=5, width=15, units ='in', dpi=300)

def plot_gene_loading(asap_adata,top_n=3,max_thresh=100):
	df_beta = pd.DataFrame(asap_adata.varm['beta'].T)
	df_beta.columns = asap_adata.var.index.values
	df_beta = df_beta.loc[:, ~df_beta.columns.duplicated(keep='first')]
	df_top = get_topic_top_genes(df_beta.iloc[:,:],top_n)
	df_top['Proportion'] = df_top['Proportion'].astype(float)
	df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
	if max_thresh:
		df_top[df_top>max_thresh] = max_thresh
	sns.clustermap(df_top.T,cmap='viridis')
	plt.savefig(asap_adata.uns['inpath']+'_beta'+'_th_'+str(max_thresh)+'.png');plt.close()

def plot_blockwise_totalgene_bp(mtx,outfile,mode,ngroups=5):

	if mode == 'var':
		data = mtx.var(0)
	elif mode == 'mean':
		data = mtx.mean(0)
	maxv = data.max()
	minv = data.min()
	widthv = (maxv - minv)/ngroups

	x = []
	for i in range(5):
		if i ==0:
			start = minv
			end = minv+widthv
		else:
			start = x[i-1][1]
			end = start+widthv
		x.append([start,end])

	block_counts = []

	for i in x:
		
		block_start = i[0]
		block_end = i[1]

		block = data[(data > block_start) & (data <= block_end)]
		block_counts.append(len(block))

	dfp = pd.DataFrame(block_counts)
	dfp.index = [v[1] for v in x]

	dfp = dfp.reset_index()
	dfp.columns = ['index','value']

	dfp['index'] = [x[:6] for x in dfp['index'].astype(str) ]
	p = ggplot(dfp, aes(x='index', y='value')) + geom_bar(stat = "identity", position = "dodge")
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white')
	) + labs(title='Gene mean var analysis', x='Gene '+mode, y='Number of genes')
	p.save(filename = outfile+'_blockwise_totalgene_bp.png', height=6, width=8, units ='in', dpi=300)

def plot_blockwise_totalgene_depth_sp(mtx,outfile,mode,ngroups=5):

	if mode == 'var':
		data = mtx.var(0)
	elif mode == 'mean':
		data = mtx.mean(0)
	maxv = data.max()
	minv = data.min()
	widthv = (maxv - minv)/ngroups

	x = []
	for i in range(5):
		if i ==0:
			start = minv
			end = minv+widthv
		else:
			start = x[i-1][1]
			end = start+widthv
		x.append([start,end])

	block_index_list = []

	for i in x:
		
		block_start = i[0]
		block_end = i[1]
		block_index = np.where((data > block_start) & (data <= block_end))
		block_index_list.append(block_index)

	bmap = {}
	depth = mtx.sum(1)
	for bi,b in enumerate(block_index_list):
		bmap[bi]={}
		bmap[bi]['mean'] = mtx[:,b[0]].mean(1)
		bmap[bi]['depth'] = depth

	df = pd.DataFrame.from_dict(bmap, orient='index')
	df = df.reset_index()

	fig, ax = plt.subplots(df.shape[0],1,figsize=(8,15))
	for i in range(df.shape[0]):
		dfp = df.iloc[i,:]
		sns.scatterplot(data=dfp, x='depth', y='mean',ax=ax[i],label=f'mean group {i}')
		ax[i].set_xticks([])
		ax[i].set_yticks([])	
	plt.savefig(outfile+'_totalgene_depth_sp.png');plt.close()

def plot_dmv_distribution(mtx,outfile):

	dfp = pd.DataFrame(mtx.sum(1),columns=['depth'])
	dfp['mean'] = mtx.mean(1)
	dfp['var'] = mtx.var(1)

	dfpm = pd.melt(dfp)

	p = ggplot(dfpm, aes(x='value')) + geom_density(alpha=0.8) + facet_wrap('~ variable', ncol=1,scales='free')
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white')
	) + ggtitle('stats')
	p.save(filename = outfile+'_dmv_dist.png', height=6, width=8, units ='in', dpi=300)

def pbulk_cellcounthist(asap_object):
	
	pblen = []
	for k,v in enumerate(asap_object.adata.uns['pseudobulk']['pb_map']):
		for in_v in asap_object.adata.uns['pseudobulk']['pb_map'][v].values():
			pblen.append(len(in_v))
	sns.histplot(x=pblen)
	plt.xlabel('Number of cells in pseudo-bulk samples')
	plt.ylabel('Number of pseudo-bulk samples')
	plt.savefig(asap_object.adata.uns['inpath']+'_pbulk_hist.png')
	plt.close()

def plot_pbulk_celltyperatio(df,outfile):

	# theme_set(theme_void())
	df = df.reset_index().rename(columns={'index': 'pbindex'})
	dfm = pd.melt(df,id_vars='pbindex')
	dfm = dfm.sort_values(['variable','value'])

	p = (ggplot(dfm, aes(x='pbindex', y='value',fill='variable')) + 
		geom_bar(position="stack",stat="identity",size=0) +
		scale_fill_manual(values=custom_palette) 		)
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white')
	)
	p.save(filename = outfile+'_pbulk_ratio.png', height=4, width=8, units ='in', dpi=300)
	

# def plot_marker_genes(fn,df,umap_coords,marker_genes,nr,nc):

# 	from anndata import AnnData
# 	import scanpy as sc
# 	import numpy as np

# 	import matplotlib.pylab as plt
# 	plt.rcParams['figure.figsize'] = [15, 10]
# 	plt.rcParams['figure.autolayout'] = True
# 	import seaborn as sns

# 	adata = AnnData(df.to_numpy())
# 	sc.pp.normalize_total(adata, target_sum=1e4)
# 	sc.pp.log1p(adata)
# 	dfn = adata.to_df()
# 	dfn.columns = df.columns
# 	dfn['cell'] = df.index.values

# 	dfn['umap1']= umap_coords[:,0]
# 	dfn['umap2']= umap_coords[:,1]

# 	fig, ax = plt.subplots(nr,nc) 
# 	ax = ax.ravel()

# 	for i,g in enumerate(marker_genes):
# 		if g in dfn.columns:
# 			print(g)
# 			val = np.array([x if x<3 else 3.0 for x in dfn[g]])
# 			sns.scatterplot(data=dfn, x='umap1', y='umap2', hue=val,s=.1,palette="viridis",ax=ax[i],legend=False)

# 			# norm = plt.Normalize(val.min(), val.max())
# 			# sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
# 			# sm.set_array([])

# 			# cax = fig.add_axes([ax[i].get_position().x1, ax[i].get_position().y0, 0.01, ax[i].get_position().height])
# 			# fig.colorbar(sm,ax=ax[i])
# 			# ax[i].axis('off')

# 			ax[i].set_title(g)
# 	fig.savefig(fn+'_umap_marker_genes_legend.png',dpi=600);plt.close()

