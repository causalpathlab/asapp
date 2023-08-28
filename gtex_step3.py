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
pyliger.leiden_cluster(ifnb_liger, resolution=0.01,k=10)

pyliger.run_umap(ifnb_liger, distance = 'correlation', n_neighbors = 100, min_dist = 0.1)

all_plots = pyliger.plot_by_dataset_and_cluster(ifnb_liger, axis_labels = ['UMAP 1', 'UMAP 2'], return_plots = True)

# List of file names to save the plots
file_names = [outpath+'plot1.png', outpath+'plot2.png']

# You can also use different formats like 'pdf', 'svg', etc.
file_formats = ['png', 'png']
for i, plot in enumerate(all_plots):
    # Construct the full file path including the format
    file_path = file_names[i]
    
    # Save the plot to the specified file path and format
    plot.save(filename=file_path, format=file_formats[i])





ifnb_liger.adata_list[0].write(outpath+'pb_liger.h5')
ifnb_liger.adata_list[1].write(outpath+'bulk_liger.h5')

dfliger0 = pd.DataFrame(ifnb_liger.adata_list[0].obsm['umap_coords'])
dfliger0['celltype'] = 'asap_pb'

dfliger1 = pd.DataFrame(ifnb_liger.adata_list[1].obsm['umap_coords'])
dfliger1['celltype'] = [x.split('@')[1] for x in ifnb_liger.adata_list[1].obs.index.values] 

dfliger = pd.concat([dfliger0,dfliger1])
dfliger.columns = ['umap1','umap2','cell-type']
import assapy
asappy.plot_umap_df(dfliger,'cell-type',outpath)