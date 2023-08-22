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
             rand_seed=42):
	
	import umap
	
	umap_coords = umap.UMAP(n_components=k, metric=distance,
						n_neighbors=n_neighbors, min_dist=min_dist,
						random_state=rand_seed).fit_transform(asap_adata.obsm[mode])

	asap_adata.obsm['umap_coords'] =  asap_adata.obsm['umap_coords'] = umap_coords	


def pmf2topic(beta, theta, eps=1e-8):
    uu = np.maximum(np.sum(beta, axis=0), eps)
    beta = beta / uu

    prop = theta * uu 
    zz = np.maximum(np.sum(prop, axis=1), eps)
    prop = prop / zz[:, np.newaxis]

    return {'beta': beta, 'prop': prop, 'depth': zz}



