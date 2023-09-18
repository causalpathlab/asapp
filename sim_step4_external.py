import sys
import numpy as np
import asappy
import pandas as pd
import scanpy as sc
import anndata as an
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



def _ligerpipeline(mtx,var,obs,outpath):
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
    pyliger.leiden_cluster(ifnb_liger, resolution=0.1,k=10)

    df1 = ifnb_liger.adata_list[0].obs[['cell','cluster']]
    df2 = ifnb_liger.adata_list[1].obs[['cell','cluster']]
    df = pd.concat([df1,df2])
    df.to_csv(outpath+'_liger.csv.gz',index=False, compression='gzip')
    

def _scanpy(adata):

        sc.tl.pca(adata, svd_solver='arpack')
        # sc.pl.pca(adata)
        # plt.savefig('_scanpy_raw_pipeline_pca.png');plt.close()

        # sc.pl.pca_variance_ratio(adata, n_pcs=50,log=True)
        # plt.savefig('_scanpy_raw_pipeline_pca_var.png');plt.close()

        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        # sc.pl.umap(adata, color=['leiden'])
        # plt.savefig('_scanpy_raw_pipeline_umap.png');plt.close()

        df_leiden = pd.DataFrame(adata.obs['leiden']) 
        df_leiden.index=adata.obs[0]
        df_leiden = df_leiden.reset_index()
        df_leiden.columns = ['cell','cluster']
        df_leiden.to_csv(outpath+'_scanpy.csv.gz',index=False, compression='gzip')


def _pca(adata,n,nc):

        if adata.shape[1] < nc:
                df_pca = pd.DataFrame()
                df_pca['cell'] = ['error']
                df_pca['cluster'] = [0]
                df_pca.to_csv(outpath+'_pc'+str(nc)+'n'+str(n)+'.csv.gz',index=False, compression='gzip')
                print('PCA-error-->')
                print(outpath)
                print(n)
                print(nc)

        else:      
                pca = PCA(n_components=nc)

                df_pca = pd.DataFrame(pca.fit_transform(adata.to_df().iloc[:,:n]))
                
                df_pca['cell'] = adata.obs[0].values

                scaler = StandardScaler()
                scaled = scaler.fit_transform(df_pca.iloc[:,:-1].to_numpy())

                kmeans = KMeans(n_clusters=K, init='k-means++',random_state=0).fit(scaled)
                df_pca['cluster'] = kmeans.labels_
                df_pca = df_pca[['cell','cluster']]

                df_pca.to_csv(outpath+'_pc'+str(nc)+'n'+str(n)+'.csv.gz',index=False, compression='gzip')



sample = sys.argv[1]
# sample = 'sim_r_0.9_s_100_sd_1'
print(sample)

data_size = 25000
number_batches = 5
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)


K = 13

# mtx=asap_object.adata.X
# var = asap_object.adata.var.values.flatten()
# obs = asap_object.adata.obs.values.flatten()
# outpath = asap_object.adata.uns['inpath']


df = asap_object.adata.construct_batch_df(asap_object.adata.uns['shape'][0])
var = df.columns.values
obs = df.index.values
mtx=df.to_numpy()
outpath = asap_object.adata.uns['inpath']


######## full external nmf model 

_ligerpipeline(mtx,var,obs,outpath)


 ### pca and scanpy pipeline


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
sc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var.highly_variable]


########

_pca(adata,10,2)
_pca(adata,100,10)
_pca(adata,1000,50)
_scanpy(adata)