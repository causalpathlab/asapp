import pandas as pd
import numpy as np
import warnings
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc
warnings.simplefilter(action='ignore', category=FutureWarning)


def pbulk_hist(asap):
	if len(asap.pbulkd) ==1 :
		sns.histplot(x=[len(asap.pbulkd[x])for x in asap.pbulkd.keys()])
		plt.savefig(asap.adata.outpath+'_pbulk_hist_full.png')
		plt.close()
	else:
		pblen = []
		for k,v in enumerate(asap.pbulkd):
			for in_v in asap.pbulkd[v].values():
				pblen.append(len(in_v))
		sns.histplot(x=pblen)
		plt.savefig(asap.adata.outpath+'_pbulk_hist_batch.png')
		plt.close()

def plot_pbulk_batchratio(df,outfile):

	from plotnine import (
		ggplot,
		aes,
		geom_bar,
		theme,
		theme_set,
		theme_void,
		element_rect
	)

	# theme_set(theme_void())
	df = df.reset_index().rename(columns={'index': 'pbindex'})
	dfm = pd.melt(df,id_vars='pbindex')
	dfm = dfm.sort_values(['variable','value'])

	p = ggplot(dfm, aes(x='pbindex', y='value',fill='variable')) + geom_bar(position="stack",stat="identity",size=0)
	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white')
	)
	p.save(filename = outfile, height=3, width=8, units ='in', dpi=300)


def generate_gene_vals(df,top_n,top_genes,label):

	top_genes_collection = []
	for x in range(df.shape[0]):
		gtab = df.T.iloc[:,x].sort_values(ascending=False)[:top_n].reset_index()
		gtab.columns = ['gene','val']
		genes = gtab['gene'].values
		for g in genes:
			if g not in top_genes_collection:
				top_genes_collection.append(g)

	for g in top_genes_collection:
		for i,x in enumerate(df[g].values):
			top_genes.append(['k'+str(i),label,'g'+str(i+1),g,x])

	return top_genes

def get_topic_top_genes(df_beta,top_n):

	top_genes = []
	top_genes = generate_gene_vals(df_beta,top_n,top_genes,'top_genes')

	return pd.DataFrame(top_genes,columns=['Topic','GeneType','Genes','Gene','Proportion'])

def get2dprojection(mtx):

	import umap

	um = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.4,metric='cosine')
	um.fit(mtx)
	return um.embedding_[:,[0,1]]



def plot_umaps(df,outpath):

	import matplotlib.pylab as plt
	plt.rcParams['figure.figsize'] = [10, 15]
	plt.rcParams['figure.autolayout'] = True
	import seaborn as sns

	import re


	labels = [ x for x in df.columns if not re.search(x,r'umap_1|umap_2|cell')]

	n_plots = len(labels)

	fig, ax = plt.subplots(n_plots,1) 
	ax = ax.ravel()

	for i,label in enumerate(labels):
		cp = sns.color_palette(cc.glasbey_hv, n_colors=len(df[label].unique()))
		sns.scatterplot(data=df, x='umap_1', y='umap_2', hue=label,s=5,palette=cp,legend=True,ax=ax[i])
		ax[i].legend(title=label,title_fontsize=12, fontsize=10,loc='center left', bbox_to_anchor=(1, 0.5))
	fig.savefig(outpath,dpi=600);plt.close()


def pmf2topic(beta, theta, eps=1e-8):
    uu = np.maximum(np.sum(beta, axis=0), eps)
    beta = beta / uu

    prop = theta * uu 
    zz = np.maximum(np.sum(prop, axis=1), eps)
    prop = prop / zz[:, np.newaxis]

    return {'beta': beta, 'prop': prop, 'depth': zz}



def plot_marker_genes(fn,mtx,rows,cols,df_umap,marker_genes):

	from anndata import AnnData
	import scanpy as sc
	import numpy as np

	import matplotlib.pylab as plt
	plt.rcParams['figure.figsize'] = [10, 5]
	plt.rcParams['figure.autolayout'] = True
	import seaborn as sns

	adata = AnnData(mtx)
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)
	dfn = adata.to_df()
	dfn.columns = cols
	dfn['cell'] = rows

	dfn = pd.merge(dfn,df_umap,on='cell',how='left')

	fig, ax = plt.subplots(2,3) 
	ax = ax.ravel()

	for i,g in enumerate(marker_genes):
		if g in dfn.columns:
			print(g)
			val = np.array([x if x<3 else 3.0 for x in dfn[g]])
			sns.scatterplot(data=dfn, x='umap1', y='umap2', hue=val,s=.1,palette="viridis",ax=ax[i],legend=False)

			norm = plt.Normalize(val.min(), val.max())
			sm = plt.cm.ScalarMappable(cmap="viridis",norm=norm)
			sm.set_array([])

			# cax = fig.add_axes([ax[i].get_position().x1, ax[i].get_position().y0, 0.01, ax[i].get_position().height])
			fig.colorbar(sm,ax=ax[i])
			ax[i].axis('off')

			ax[i].set_title(g)
	fig.savefig(fn+'_umap_marker_genes_legend.png',dpi=600);plt.close()

