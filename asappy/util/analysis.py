import pandas as pd
import numpy as np

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

def run_umap(asap_adata,
	         mode = 'corr',
             k=2,
             distance="euclidean",
             n_neighbors=10,
             min_dist=0.1,
	     	 use_snn = True,
             rand_seed=42):

	if use_snn:

		from umap.umap_ import find_ab_params, simplicial_set_embedding
		
		n_components = k
		spread: float = 1.0
		alpha: float = 1.0
		gamma: float = 1.0
		negative_sample_rate: int = 5
		maxiter = None
		default_epochs = 500 if asap_adata.obsp['snn'].shape[0] <= 10000 else 200
		n_epochs = default_epochs if maxiter is None else maxiter
		random_state = np.random.RandomState(rand_seed)

		a, b = find_ab_params(spread, min_dist)

		umap_coords = simplicial_set_embedding(
			data = asap_adata.obsm[mode],
			graph = asap_adata.obsp['snn'],
			n_components=n_components,
			initial_alpha = alpha,
			a = a,
			b = b,
			gamma = gamma,
			negative_sample_rate = negative_sample_rate,
			n_epochs = n_epochs,
			init='spectral',
			random_state = random_state,
			metric = distance,
			metric_kwds = {},
			densmap=False,
			densmap_kwds={},
			output_dens=False
			)
		asap_adata.obsm['umap_coords'] = umap_coords[0]

	else:
		import umap
		
		umap_coords = umap.UMAP(n_components=k, metric=distance,
							n_neighbors=n_neighbors, min_dist=min_dist,
							random_state=rand_seed).fit_transform(asap_adata.obsm[mode])

		asap_adata.obsm['umap_coords'] = umap_coords	

def get_psuedobulk_batchratio(asap_object,batch_label):

	pb_batch_count = []
	batch_label = np.array(batch_label)
	batches = set(batch_label)

	for _,pb_map in asap_object.adata.uns['pseudobulk']['pb_map'].items():
		for _,val in pb_map.items():
			pb_batch_count.append([np.count_nonzero(batch_label[val]==x) for x in batches])
		
	df_pb_batch_count = pd.DataFrame(pb_batch_count).T
	df_pb_batch_count = df_pb_batch_count.T
	df_pb_batch_count.columns = batches
	pbulk_batchratio = df_pb_batch_count.div(df_pb_batch_count.sum(axis=1), axis=0)
	
	return pbulk_batchratio


def pmf2topic(beta, theta, eps=1e-8):
    uu = np.maximum(np.sum(beta, axis=0), eps)
    beta = beta / uu

    prop = theta * uu 
    zz = np.maximum(np.sum(prop, axis=1), eps)
    prop = prop / zz[:, np.newaxis]

    return {'beta': beta, 'prop': prop, 'depth': zz}


def row_col_order(dfm):

	from scipy.cluster import hierarchy

	df = dfm.copy()
 
	Z = hierarchy.ward(df.to_numpy())
	ro = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, df.to_numpy()))

	df['topic'] = df.index.values
	dfm = pd.melt(df,id_vars='topic')
	dfm.columns = ['row','col','values']
	M = dfm[['row', 'col', 'values']].copy()
	M['row'] = pd.Categorical(M['row'], categories=ro)
	M = M.pivot(index='row', columns='col', values='values').fillna(0)
	co = np.argsort(-M.values.max(axis=0))
	co = M.columns[co]
 
	return ro,co

