import sys
import numpy as np
from asap.data.dataloader import DataSet
import asapc
import pandas as pd
import scanpy as sc
import anndata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



inpath = sys.argv[1]
outpath = sys.argv[2]

dl = DataSet(inpath,outpath,data_mode='sparse',data_ondisk=False)
dl.initialize_data()
dl.load_data()


K = 25

######## full external nmf model 

print('full external nmf model...scanpy ')


 ### get high variable genes


adata = anndata.AnnData(dl.mtx)
dfvars = pd.DataFrame(dl.cols)
dfobs = pd.DataFrame(dl.rows)
adata.obs = dfobs
adata.var = dfvars

sc.pp.filter_cells(adata, min_genes=0)
sc.pp.filter_genes(adata, min_cells=0)
adata.var['mt'] = adata.var_names.str.startswith('MT-')  
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pp.normalize_total(adata, target_sum=10000)

sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]


########

def _scanpy(adata):
        sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
        sc.pp.scale(adata, max_value=10)

        sc.tl.pca(adata, n_comps=K, svd_solver='arpack')

        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)

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


_pca(adata,25,25)
_pca(adata,50,25)
_pca(adata,500,25)
_scanpy(adata)
