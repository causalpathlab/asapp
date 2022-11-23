import pandas as pd
import numpy as np

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
