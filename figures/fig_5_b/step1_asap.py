from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import asappy
import pandas as pd
import numpy as np
import anndata as an

########################################

import scanpy as sc
adata = sc.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
adata.var_names_make_unique()
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)



sc.pp.filter_cells(adata, min_counts=1000)
sc.pp.filter_cells(adata, max_counts=50000)
adata = adata[adata.obs["pct_counts_mt"] < 20]
print(f"#cells after MT filter: {adata.n_obs}")
sc.pp.filter_genes(adata, min_cells=2)


import h5py as hf
from scipy.sparse import csr_matrix
df = adata.to_df()
smat = csr_matrix(df.to_numpy())
barcodes = df.index.values
genes = list(df.columns)

fname='data/brca_sp'
f = hf.File(fname+'.h5','w')
grp = f.create_group('matrix')
grp.create_dataset('barcodes', data = barcodes ,compression='gzip')
grp.create_dataset('indptr',data=smat.indptr,compression='gzip')
grp.create_dataset('indices',data=smat.indices,compression='gzip')
grp.create_dataset('data',data=smat.data,compression='gzip')
data_shape = np.array([len(barcodes),len(genes)])
grp.create_dataset('shape',data=data_shape)
f['matrix'].create_group('features')
f['matrix']['features'].create_dataset('id',data=genes,compression='gzip')
f.close()


########################################
######################################################
import sys 
sample = 'brca_sp'

wdir = 'experiments/asapp/figures/fig_5_b/'

data_size = 10000
number_batches = 1


# asappy.create_asap_data(sample,working_dirpath=wdir)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)
   
def analyze_nmf():

    asappy.generate_pseudobulk(asap_object,tree_depth=10,downsample_pseudobulk=False,pseudobulk_filter=False)
    
    n_topics = 7 ## paper 
    asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
    
    asap_adata = asappy.generate_model(asap_object,return_object=True)
    asappy.generate_model(asap_object)
    
    asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')
    
    cluster_resolution= 0.6 ## paper
    asappy.leiden_cluster(asap_adata,resolution=cluster_resolution)
    print(asap_adata.obs.cluster.value_counts())
    
    ## min distance 0.5 paper
    asappy.run_umap(asap_adata,distance='euclidean',min_dist=0.1)
    asappy.plot_umap(asap_adata,col='cluster',pt_size=2.5,ftype='png')
    asappy.plot_umap(asap_adata,col='cluster',pt_size=2.5,ftype='pdf')

    pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
    theta = pd.DataFrame(pmf2t['prop'])
    theta.columns = ['t'+str(x) for x in theta.columns]

    asap_adata.write_h5ad(wdir+'results/'+sample+'.h5asapad')

    return asap_adata,theta
    
# 	df = pd.DataFrame(asap_adata.obsm['corr'])
    # cl = asap_adata.obs.index.values
    # ct, ct2 = getct(cl,sample,minor=True)
    # asap_adata.obs['celltype']  = pd.Categorical(ct)
    # asap_adata.obs['celltype_m']  = pd.Categorical(ct2)
    # asappy.plot_umap(asap_adata,col='celltype',pt_size=0.5,ftype='png')
    # asappy.plot_umap(asap_adata,col='celltype_m',pt_size=0.5,ftype='png')
    
    # asap_adata.write(wdir+'results/'+sample+'.h5asapad')
    
    # ## top 10 main paper
    # asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=25)
    
    # asap_s1,asap_s2 =  calc_score([str(x) for x in asap_adata.obs['celltype'].values],asap_adata.obs['cluster'].values)

    # print('---NMF-----')
    # print('NMI:'+str(asap_s1))
    # print('ARI:'+str(asap_s2))


asap_adata,theta = analyze_nmf()


sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)


sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="clusters")


# adata.write_h5ad('results/_scanpy.h5asap')

import pandas as pd
import anndata as an
sample = 'brca_sp'

wdir = 'experiments/asapp/figures/fig_5_b/'

adata = an.read_h5ad('results/_scanpy.h5asap')
asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')

sc.pl.umap(adata, color=["total_counts", "n_genes_by_counts", "clusters"], wspace=0.4)

import matplotlib.pylab as plt
 

plt.savefig(asap_adata.uns['inpath']+'scanpy_cluster.png');plt.close()

sc.pl.spatial(adata, img_key="hires", color=["clusters"])

plt.savefig(asap_adata.uns['inpath']+'scanpy_spatial.png');plt.close()

adata.obs['asap'] = pd.Categorical(asap_adata.obs['cluster'].values )
sc.pl.spatial(adata, img_key="hires", color=["asap"])
plt.savefig(asap_adata.uns['inpath']+'asap_spatial.pdf');plt.close()

#####################################
##########draw spatial plots 
#####################################


# plt.rcParams['figure.figsize'] = [15, 10]
# plt.rcParams['figure.autolayout'] = True

# nr =2
# nc = 5
# fig, ax = plt.subplots(nr,nc) 
# ax = ax.ravel()
# for i,t in enumerate(theta.columns):
#     adata.obs[t] = theta[t].values 
#     sc.pl.spatial(adata, img_key="hires", color=[t],ax=ax[i])
#     ax[i].set_title(t)
#     ax[i].legend().set_visible(False)
    
# fig.savefig(asap_adata.uns['inpath']+'asap_spatial_topic.png',dpi=600);plt.close()


# import anndata as an
# from sklearn.preprocessing import MinMaxScaler
# asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')

# asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=25)



pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
df = pd.DataFrame(pmf2t['prop'])

from sklearn.preprocessing import StandardScaler
df = asap_adata.obsm['corr']
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df))
# scaler = MinMaxScaler()
# df = pd.DataFrame(scaler.fit_transform(df))
df.columns = ['t'+str(x) for x in df.columns]
df[df>3]=3
df[df<0.5]=0

import matplotlib.pylab as plt
import seaborn as sns
import scanpy as sc

sns.clustermap(df,cmap='viridis')
plt.savefig(asap_adata.uns['inpath']+'_prop_hmap.png',dpi=600);plt.close()



plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['figure.autolayout'] = True

nr =2
nc = 4
fig, ax = plt.subplots(nr,nc) 
ax = ax.ravel()
for i,t in enumerate(df.columns):
    adata.obs[t] = df[t].values 
    sc.pl.spatial(adata, img_key="hires", color=[t],ax=ax[i],cmap='Oranges')
    ax[i].set_title(t)
    ax[i].legend().set_visible(False)
    
fig.savefig(asap_adata.uns['inpath']+'asap_spatial_topic_3.pdf',dpi=600);plt.close()




####### pseudobulk analysis

asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asap')

pb_coord = []
for pbk in asap_adata.uns['pseudobulk']['pb_map']['1_0_3658'].keys():
    pb = asap_adata.uns['pseudobulk']['pb_map']['1_0_3658'][pbk]
    if pb.size >= 5:
        pb_coord.append(np.median(adata.obsm['spatial'][pb],axis=0))
    
'''
total len is 600
if >5 then 218 pb samples

'''

pb_coord = np.array(pb_coord).astype(int)

pbsize = pb_coord.shape[0]
adata.obsm['spatial'][:pbsize] = pb_coord
pb_mark = [str(x) for x in range(adata.obs.shape[0])]

adata.obs['pb_mark'] = pb_mark
adata.obs['pb_mark'][:pbsize] = 'pb'
adata.obs['pb_mark'][pbsize:] = 'not_pb'
sc.pl.spatial(adata, img_key="hires", color=["pb_mark"],palette=['grey','#FF6103'],spot_size=400,groups=['pb'],cmap='Oranges')
plt.savefig(asap_adata.uns['inpath']+'pb_spatial.pdf');plt.close()
# plt.savefig(asap_adata.uns['inpath']+'pb_spatial.png');plt.close()
