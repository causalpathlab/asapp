######################################################
##### single cell nmf
######################################################
import asappy
import sys


sample = str(sys.argv[1])
n_topics = int(sys.argv[2])
cluster_resolution = float(sys.argv[3])
wdir = sys.argv[4]
print(sample)

asappy.create_asap_data(sample=sample,working_dirpath=wdir)

data_size = 25000
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)
asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale')
asappy.asap_nmf(asap_object,num_factors=n_topics)
asappy.generate_model(asap_object)



import pandas as pd

def getct(ids,sample):
    
    if 'sim' in sample:
        ct = [ x.replace('@'+ sample,'') for x in ids]
        ct = [ '-'.join(x.split('_')[1:]) for x in ct]
        return ct 
    else:
        dfid = pd.DataFrame(ids,columns=['cell'])
        dfl = pd.read_csv(wdir+'results/'+sample.split('_')[0]+'_celltype.csv.gz')
        dfjoin = pd.merge(dfl,dfid,on='cell',how='right')
        ct = dfjoin['celltype'].values
        
        return ct


import asappy
import anndata as an
from pyensembl import ensembl_grch38

asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')

# gn = []
# for x in asap_adata.var.index.values:
#     try:
#         g = ensembl_grch38.gene_by_id(x.split('.')[0]).gene_name 
#         gn.append(g)
#     except:
#         gn.append(x)

# asap_adata.var.index = gn


##### beta heatmap
# asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=30)


##### cluster and celltype umap
asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
# print(asap_adata.obs.cluster.value_counts())
asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
asappy.plot_umap(asap_adata,col='cluster')
asap_adata.obs['celltype'] = getct(asap_adata.obs.index.values,sample)
asappy.plot_umap(asap_adata,col='celltype')
asap_adata.write(wdir+'results/'+sample+'.h5asapad')