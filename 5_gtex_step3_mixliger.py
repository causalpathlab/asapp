# ######################################################
# ##### bulk setup
# ######################################################

sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'+sample

# ######################################################
##### mix analysis
######################################################

### pyliger

import anndata as an
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler


scadata = an.read_h5ad('./results/gtex_sc.h5asap')

### bulk raw data
dfbulk = pd.read_csv(outpath+'.csv.gz')
dfbulk.set_index('Unnamed: 0',inplace=True)
dfbulk = dfbulk.astype(float)
dfbulk  = dfbulk.div(dfbulk.sum(1),axis=0) * 1e4

dfpb = pd.DataFrame(scadata.uns['pseudobulk']['pb_data']).T
dfpb.columns = dfbulk.columns
dfpb = dfpb.astype(float)
dfpb.index = ['pb-'+str(x) for x in range(dfpb.shape[0])]
dfpb  = dfpb.div(dfpb.sum(1),axis=0) * 1e4

pbadata = an.AnnData(dfpb, 
        dfpb.index.to_frame(), 
        dfpb.columns.to_frame())
pbadata.var.rename(columns={0:'gene'},inplace=True) 
pbadata.obs.rename(columns={0:'cell'},inplace=True) 
pbadata.obs.index.name = 'cell'
pbadata.var.index.name = 'gene'
pbadata.uns['sample_name'] = 'pb'

blkadata = an.AnnData(dfbulk, 
        dfbulk.index.to_frame(), 
        dfbulk.columns.to_frame())
blkadata.var.rename(columns={0:'gene'},inplace=True) 
blkadata.obs.rename(columns={0:'cell'},inplace=True) 
blkadata.obs.index.name = 'cell'
blkadata.var.index.name = 'gene'
blkadata.uns['sample_name'] = 'bulk'


adata_list = [pbadata, blkadata]

cd /home/BCCRC.CA/ssubedi/projects/experiments/liger/src
import pyliger

ifnb_liger = pyliger.create_liger(adata_list)

pyliger.normalize(ifnb_liger)
pyliger.select_genes(ifnb_liger)
pyliger.scale_not_center(ifnb_liger)
pyliger.optimize_ALS(ifnb_liger, k = 10)
pyliger.quantile_norm(ifnb_liger)
# pyliger.leiden_cluster(ifnb_liger, resolution=0.01,k=10)

df1 = pd.DataFrame(ifnb_liger.adata_list[0].obsm['H_norm'])
df1.index = [str(x)+'pb' for x in df1.index.values]
df2 = pd.DataFrame(ifnb_liger.adata_list[1].obsm['H_norm'])
df2.index = [str(x)+'bulk' for x in df2.index.values]
df3 = pd.concat([df1,df2])

# df3.to_csv(outpath+'liger_h_norm.csv.gz',compression='gzip')

# pyliger.run_umap(ifnb_liger, distance = 'euclidean', n_neighbors = 25, min_dist = 0.01)

# all_plots = pyliger.plot_by_dataset_and_cluster(ifnb_liger, axis_labels = ['UMAP 1', 'UMAP 2'], return_plots = True)

# # List of file names to save the plots
# file_names = [outpath+'plot12.png', outpath+'plot22.png']

# # You can also use different formats like 'pdf', 'svg', etc.
# file_formats = ['png', 'png']
# for i, plot in enumerate(all_plots):
#     # Construct the full file path including the format
#     file_path = file_names[i]
    
#     # Save the plot to the specified file path and format
#     plot.save(filename=file_path, format=file_formats[i])





# ifnb_liger.adata_list[0].write(outpath+'pb_liger.h5')
# ifnb_liger.adata_list[1].write(outpath+'bulk_liger.h5')

# dfliger0 = pd.DataFrame(ifnb_liger.adata_list[0].obsm['umap_coords'])
# dfliger0['celltype'] = 'asap_pb'

# dfliger1 = pd.DataFrame(ifnb_liger.adata_list[1].obsm['umap_coords'])
# dfliger1['celltype'] = [x.split('@')[1] for x in ifnb_liger.adata_list[1].obs.index.values] 

# dfliger = pd.concat([dfliger0,dfliger1])
# dfliger.columns = ['umap1','umap2','cell-type']

# import asappy
# df=pd.read_csv(outpath+'liger_h_norm_umap.csv')
# df.columns = ['cell','umap1','umap2']
# df['batch'] = ['pb' if 'pb' in x else 'bulk' for x in df['cell']]


# df['cluster'] = list(ifnb_liger.adata_list[0].obs['cluster'].values) + list(ifnb_liger.adata_list[1].obs['cluster'].values)

# df['cluster'] = pd.Categorical(df['cluster'])

# asappy.plot_umap_df(df,'batch',outpath)
# asappy.plot_umap_df(df,'cluster',outpath)




snn,cluster = leiden_cluster(df3,resolution=0.1,k=25)

from umap.umap_ import find_ab_params, simplicial_set_embedding

min_dist = 0.8
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
        data = df3.to_numpy(),
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

df = pd.DataFrame(umap_coords[0])
df['cell'] = df3.index.values
df.columns = ['umap1','umap2','cell']
df['batch'] = ['pb' if 'pb' in x else 'bulk' for x in df['cell']]
df['cluster'] = cluster
df['cluster'] = pd.Categorical(df['cluster'])
df['tissue'] = ['a' for x in dfpb.index.values] + [x.split('@')[1] for x in dfbulk.index.values]
# outpath +='test'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/mix_'
plot_umap_df(df,'batch',outpath)
plot_umap_df(df,'cluster',outpath)
plot_umap_df(df,'tissue',outpath)

dfh = df[['cell','cluster','tissue']]
dfh = dfh.groupby(['cluster','tissue'])['cell'].count().reset_index().sort_values(['cluster','cell'])
dfh = dfh.pivot(index='tissue',columns='cluster',values='cell')
dfh =  dfh.div(dfh.sum(0))
dfh.rename(index={'a':'pseudo-bulk'},inplace=True)
dfh.at['a','Name'] = 'pseudo-bulk'

import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6)) 

sns.heatmap(dfh,cmap='viridis')
plt.tight_layout()
plt.savefig(outpath+'mix_hmap.png');plt.close()



# ######################################################
# ##### correlation analysis between bulk and pb
# ######################################################

sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'+sample

# ######################################################
##### mix analysis
######################################################
import anndata as an
import pandas as pd
import numpy as np

scadata = an.read_h5ad('./results/gtex_sc.h5asap')

model = np.load(outpath+'_nmf.npz',allow_pickle=True)
bulkadata = an.AnnData(shape=(model['theta'].shape[0],model['beta'].shape[0]))
bulkadata.obs_names = [ 'b'+str(x) for x in range(model['theta'].shape[0])]
bulkadata.var_names = pd.read_csv('./results/bulkcommon_genes.csv').values.flatten()
bulkadata.varm['beta'] = model['beta']
bulkadata.obsm['theta'] = model['theta']
bulkadata.obsm['corr'] = model['corr']
bulkadata.uns['inpath'] = outpath

n_genes = 5
sc_beta = pd.DataFrame(scadata.varm['beta'].T)
sc_beta.columns = scadata.var.index.values
bulk_beta = pd.DataFrame(bulkadata.varm['beta'].T)
bulk_beta.columns = scadata.var.index.values

from asappy.util import analysis
sc_top_genes = np.unique(analysis.get_topic_top_genes(sc_beta,n_genes)['Gene'].values)
bulk_top_genes = np.unique(analysis.get_topic_top_genes(bulk_beta,n_genes)['Gene'].values)

top_genes = np.unique(list(sc_top_genes)+list(bulk_top_genes))
top_genes = bulk_top_genes

sc_beta = sc_beta[top_genes]
bulk_beta = bulk_beta[top_genes]

sc_beta.index = ['pb_'+str(x) for x in sc_beta.index.values]
bulk_beta.index = ['bulk_'+str(x) for x in bulk_beta.index.values]


correlation_results = pd.DataFrame(index=sc_beta.index, columns=bulk_beta.index)
for i, row1 in sc_beta.iterrows():
    for j, row2 in bulk_beta.iterrows():
        correlation_results.loc[i, j] = row1.corr(row2)
correlation_results = correlation_results.apply(pd.to_numeric)
sns.clustermap(correlation_results,cmap='viridis')
plt.savefig(outpath+'_correlation_mix.png');plt.close()


beta = pd.concat([sc_beta,bulk_beta])

import seaborn as sns
import matplotlib.pyplot as plt

sns.clustermap(sc_beta.T,cmap='viridis',col_cluster=False)
plt.savefig(outpath+'_betamix.png');plt.close()
