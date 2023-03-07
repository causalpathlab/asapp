import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_gene_vals(df,top_n,top_genes,label):

	top_genes_collection = []
	for x in range(df.shape[0]):
		gtab = df.T.iloc[:,x].sort_values(0,ascending=False)[:top_n].reset_index()
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

