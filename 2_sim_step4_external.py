import sys
import numpy as np
import asappy
import pandas as pd
import scanpy as sc
import anndata as an
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def kmeans_cluster(df,k):
        kmeans = KMeans(n_clusters=k, init='k-means++',random_state=0).fit(df)
        return kmeans.labels_

def _ligerpipeline(mtx,var,obs,outpath,K):

        from sklearn.model_selection import train_test_split
        import pyliger

        adata = an.AnnData(mtx)
        dfvars = pd.DataFrame(var)
        dfobs = pd.DataFrame(obs)
        adata.obs = dfobs
        adata.var = dfvars
        adata.var.rename(columns={0:'gene'},inplace=True) 
        adata.obs.rename(columns={0:'cell'},inplace=True) 
        adata.obs.index.name = 'cell'
        adata.var.index.name = 'gene'

        test_size = 0.5
        adata_train, adata_test = train_test_split(adata, test_size=test_size, random_state=42)

        adata_train.uns['sample_name'] = 'train'
        adata_test.uns['sample_name'] = 'test'
        adata_list = [adata_train,adata_test]


        ifnb_liger = pyliger.create_liger(adata_list)

        pyliger.normalize(ifnb_liger)
        pyliger.select_genes(ifnb_liger)
        pyliger.scale_not_center(ifnb_liger)
        pyliger.optimize_ALS(ifnb_liger, k = K)
        pyliger.quantile_norm(ifnb_liger)

        # H_norm = pd.DataFrame(np.vstack([adata.obsm['H_norm'] for adata in ifnb_liger.adata_list]))
        # _,cluster = asappy.leiden_cluster(H_norm)
        # cluster = kmeans_cluster(H_norm,n_topics)

        pyliger.leiden_cluster(ifnb_liger)

        obs = list(ifnb_liger.adata_list[0].obs.cell.values) + list(ifnb_liger.adata_list[1].obs.cell.values)
        cluster = list(ifnb_liger.adata_list[0].obs.cluster.values) + list(ifnb_liger.adata_list[1].obs.cluster.values)
        H_norm = pd.DataFrame()
        
        H_norm['cell'] = obs        
        H_norm['cluster'] = cluster
        H_norm = H_norm[['cell','cluster']]
        H_norm.to_csv(outpath+'_liger.csv.gz',index=False, compression='gzip')
    

def _baseline(mtx,obs,var):
        adata = an.AnnData(mtx)
        dfvars = pd.DataFrame(var)
        dfobs = pd.DataFrame(obs)
        adata.obs = dfobs
        adata.var = dfvars

        sc.pp.filter_cells(adata, min_genes=0)
        sc.pp.filter_genes(adata, min_cells=0)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        adata = adata[:, adata.var.highly_variable]
        sc.tl.pca(adata)

        # df = pd.DataFrame(adata.obsm['X_pca'])
        # _,cluster = asappy.leiden_cluster(df)
        # cluster = kmeans_cluster(df,n_topics)
        
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        df = pd.DataFrame()        
        df['cell'] = adata.obs[0].values        
        df['cluster'] = adata.obs.leiden.values
        df = df[['cell','cluster']]
        df.to_csv(outpath+'_baseline.csv.gz',index=False, compression='gzip')


def _randomprojection(mtx,obs,depth):
        
        from asappy.projection import rpstruct
        
        rp = rpstruct.projection_data(depth,mtx.shape[1])
        rp_data = rpstruct.get_random_projection_data(mtx.T,rp)
        
        df_rp = pd.DataFrame(rp_data)
        # _,cluster = asappy.leiden_cluster(df_rp)
        cluster = kmeans_cluster(df_rp,n_topics)
        
        df_rp['cell'] = obs        
        df_rp['cluster'] = cluster
        df_rp = df_rp[['cell','cluster']]

        df_rp.to_csv(outpath+'_rp'+str(depth)+'.csv.gz',index=False, compression='gzip')

def _pca(mtx,obs,pc_n):

        pca = PCA(n_components=pc_n)
        df_pca = pd.DataFrame(pca.fit_transform(mtx))

        # _,cluster = asappy.leiden_cluster(df_pca)
        cluster = kmeans_cluster(df_pca,n_topics)
        
        df_pca['cell'] = obs
        df_pca['cluster'] = cluster
        df_pca = df_pca[['cell','cluster']]
        df_pca.to_csv(outpath+'_pc'+str(pc_n)+'.csv.gz',index=False, compression='gzip')



sample = sys.argv[1]
print(sample)

data_size = 25000
number_batches = 1
# asappy.create_asap_data(sample)
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)
n_topics = 7

current_dsize = asap_object.adata.uns['shape'][0]
# current_dsize = 25000

df = asap_object.adata.construct_batch_df(current_dsize)
var = df.columns.values
obs = df.index.values
mtx = df.to_numpy()
outpath = asap_object.adata.uns['inpath']


# print('running liger...')
######## full external nmf model 
# _ligerpipeline(mtx,var,obs,outpath,n_topics)

print('running random projection...')
######## random projection 
_randomprojection(mtx,obs,50)
_randomprojection(mtx,obs,10)
_randomprojection(mtx,obs,5)

########
print('running pca...')
_pca(mtx,obs,50)
_pca(mtx,obs,10)
_pca(mtx,obs,5)

# ######## baseline 
print('running baseline...')
_baseline(mtx,obs,var)