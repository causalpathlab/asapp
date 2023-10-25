
def run_basic_pipeline(outpath,df):
	import scanpy as sc
	import pandas as pd
	import matplotlib.pylab as plt

	adata = sc.AnnData(df, 
		df.index.to_frame(), 
		df.columns.to_frame())

	sc.pp.filter_cells(adata, min_genes=100)
	sc.pp.filter_genes(adata, min_cells=3)
	adata.var['mt'] = adata.var_names.str.startswith('MT-')  
	sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
	adata = adata[adata.obs.n_genes_by_counts < 2500, :]
	adata = adata[adata.obs.pct_counts_mt < 5, :]
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)

	sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
	adata = adata[:, adata.var.highly_variable]
	sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
	sc.pp.scale(adata, max_value=10)

	sc.tl.pca(adata, svd_solver='arpack')
	sc.pl.pca(adata)
	plt.savefig(outpath+'_scanpy_pca.png');plt.close()


	sc.pp.neighbors(adata)
	sc.tl.umap(adata)
	sc.tl.leiden(adata,resolution=0.2)
	sc.pl.umap(adata, color=['leiden'])
	plt.savefig(outpath+'_scanpy_umapLEIDEN.png');plt.close()

	df_label= pd.DataFrame(adata.obs)
	df_label = df_label.rename(columns={0:'cell'}) 
	df_label = df_label[['cell','leiden']]

	# df_leiden = pd.DataFrame(adata.obsm['X_umap']) 
	# df_leiden.columns = ['umap1','umap2']  
	# df_label['umap1'] = df_leiden['umap1'].values
	# df_label['umap2'] = df_leiden['umap2'].values

	df_label.to_csv(outpath+'_scanpy_label.csv.gz',index=False, compression='gzip')
