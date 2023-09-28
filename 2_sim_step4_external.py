import sys
import numpy as np
import asappy
import pandas as pd
import scanpy as sc
import anndata as an
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



def _ligerpipeline(mtx,var,obs,outpath,K):
    adata = an.AnnData(mtx)
    dfvars = pd.DataFrame(var)
    dfobs = pd.DataFrame(obs)
    adata.obs = dfobs
    adata.var = dfvars
    adata.var.rename(columns={0:'gene'},inplace=True) 
    adata.obs.rename(columns={0:'cell'},inplace=True) 
    adata.obs.index.name = 'cell'
    adata.var.index.name = 'gene'

    # Define the ratio for splitting (e.g., 70% train, 30% test)
    test_size = 0.5
    from sklearn.model_selection import train_test_split
    # Split the AnnData object into two random subsets
    adata_train, adata_test = train_test_split(adata, test_size=test_size, random_state=42)

    adata_train.uns['sample_name'] = 'train'
    adata_test.uns['sample_name'] = 'test'
    adata_list = [adata_train,adata_test]

    import pyliger

    ifnb_liger = pyliger.create_liger(adata_list)

    pyliger.normalize(ifnb_liger)
    pyliger.select_genes(ifnb_liger)
    pyliger.scale_not_center(ifnb_liger)
    pyliger.optimize_ALS(ifnb_liger, k = K)
    pyliger.quantile_norm(ifnb_liger)
    pyliger.leiden_cluster(ifnb_liger, resolution=0.1,k=K)

    df1 = ifnb_liger.adata_list[0].obs[['cell','cluster']]
    df2 = ifnb_liger.adata_list[1].obs[['cell','cluster']]
    df = pd.concat([df1,df2])
    df.to_csv(outpath+'_liger.csv.gz',index=False, compression='gzip')
    

def _pca(ad,pc_n):

        adata = ad.copy()
        sc.pp.highly_variable_genes(adata)
        adata = adata[:, adata.var.highly_variable]
        sc.tl.pca(adata,n_comps=pc_n ,svd_solver='arpack')
        # sc.pl.pca(adata)
        # plt.savefig('_scanpy_raw_pipeline_pca.png');plt.close()

        # sc.pl.pca_variance_ratio(adata, n_pcs=50,log=True)
        # plt.savefig('_scanpy_raw_pipeline_pca_var.png');plt.close()

        sc.pp.neighbors(adata)
        # sc.tl.umap(adata)
        sc.tl.leiden(adata)
        # sc.pl.umap(adata, color=['leiden'])
        # plt.savefig('_scanpy_raw_pipeline_umap.png');plt.close()

        df_leiden = pd.DataFrame(adata.obs['leiden']) 
        df_leiden.index=adata.obs[0]
        df_leiden = df_leiden.reset_index()
        df_leiden.columns = ['cell','cluster']
        df_leiden.to_csv(outpath+'_pc'+str(pc_n)+'.csv.gz',index=False, compression='gzip')


def _randomprojection(mtx,obs,depth):
        
        from asappy.projection import rpstruct
        rp = rpstruct.projection_data(depth,mtx.shape[1])
        rp_data = rpstruct.get_random_projection_data(mtx.T,rp)
        
        df_rp = pd.DataFrame(rp_data)
        df_rp['cell'] = obs

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_rp.iloc[:,:-1].to_numpy())

        kmeans = KMeans(n_clusters=K, init='k-means++',random_state=0).fit(scaled)
        df_rp['cluster'] = kmeans.labels_
        df_rp = df_rp[['cell','cluster']]

        df_rp.to_csv(outpath+'_rp'+str(depth)+'.csv.gz',index=False, compression='gzip')

def _baseline(adata):

        pc_n=50
        pca = PCA(n_components=pc_n)
        df_pca = pd.DataFrame(pca.fit_transform(adata.to_df()))
        df_pca['cell'] = adata.obs[0].values

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_pca.iloc[:,:-1].to_numpy())

        kmeans = KMeans(n_clusters=K, init='k-means++',random_state=0).fit(scaled)
        df_pca['cluster'] = kmeans.labels_
        df_pca = df_pca[['cell','cluster']]

        df_pca.to_csv(outpath+'_baseline.csv.gz',index=False, compression='gzip')



sample = sys.argv[1]
# sample = 'sim_r_0.0_s_10_sd_1'
print(sample)

data_size = 25000
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)


K = 13

df = asap_object.adata.construct_batch_df(asap_object.adata.uns['shape'][0])
var = df.columns.values
obs = df.index.values
mtx = df.to_numpy()
outpath = asap_object.adata.uns['inpath']


######## full external nmf model 

_ligerpipeline(mtx,var,obs,outpath,K)

######## random projection 

_randomprojection(mtx,obs,5)
_randomprojection(mtx,obs,10)
_randomprojection(mtx,obs,50)


# ######## PCA and baseline 

adata = an.AnnData(mtx)
dfvars = pd.DataFrame(var)
dfobs = pd.DataFrame(obs)
adata.obs = dfobs
adata.var = dfvars

########### common

sc.pp.filter_cells(adata, min_genes=0)
sc.pp.filter_genes(adata, min_cells=0)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)



########
_baseline(adata)
_pca(adata,50)
_pca(adata,10)
_pca(adata,5)