sample = 'pbmc'
import asappy
# asappy.create_asap_data(sample)
data_size = 25000
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

normalization_raw='unitnorm'
normalization_pb='lognorm'

asappy.generate_pseudobulk(asap_object,tree_depth=10,normalization_raw=normalization_raw,normalization_pb=normalization_pb)
asappy.asap_nmf(asap_object,num_factors=10)
asappy.save_model(asap_object)

import anndata as an
from pyensembl import ensembl_grch38
asap_adata = an.read_h5ad('./results/'+sample+'.h5asap')

gn = []
for x in asap_adata.var.index.values:
    try:
        g = ensembl_grch38.gene_by_id(x.split('.')[0]).gene_name 
        if g =='': g = x.split('.')[0]
        gn.append(g)
    except:
        gn.append(x)

asap_adata.var.index = gn

asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=10)
asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=100)
asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=1000)

asappy.leiden_cluster(asap_adata,k=10,mode='corr',resolution=0.1)
asap_adata.obs.cluster.value_counts()
asappy.run_umap(asap_adata,distance='cosine',min_dist=0.2)
asappy.plot_umap(asap_adata,col='cluster')

import pandas as pd
df = pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_results/pbmc_results/pbmc_scanpy_label.csv.gz')
df['cell'] = [x.split('@')[0] for x in df['cell']]
leiden_map = {x:y for x,y in zip(df['cell'],df['leiden'])}
asap_adata.obs['celltype'] = [leiden_map[x] if x in leiden_map.keys() else 'others' for x in asap_adata.obs.index.values]
asappy.plot_umap(asap_adata,col='celltype')
