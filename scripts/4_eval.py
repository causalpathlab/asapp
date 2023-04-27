import sys
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score


import annoy

class ApproxNN():
	def __init__(self, data, labels):
		self.dimension = data.shape[1]
		self.data = data.astype('float32')
		self.labels = labels

	def build(self, number_of_trees=50):
		self.index = annoy.AnnoyIndex(self.dimension,'angular')
		for i, vec in enumerate(self.data):
			self.index.add_item(i, vec.tolist())
		self.index.build(number_of_trees)

	def query(self, vector, k):
		indexes = self.index.get_nns_by_vector(vector.tolist(),k)
		return [self.labels[i] for i in indexes]


def get_neighbours_distances(dl_mtx):

	nbr_dist=[]
	model_ann = ApproxNN(dl_mtx,data_rows)
	model_ann.build()

	for idx,cell in enumerate(data_rows):
		nbrs,ndist = model_ann.index.get_nns_by_vector(np.asarray(dl_mtx[idx,:]).flatten(),n=len(data_rows),include_distances=True)
		nbr_dist.append(np.array(ndist)[np.array(nbrs).argsort()])

	return np.array(nbr_dist)


sim_data_path = sys.argv[1]
nmf_data_path = sys.argv[2]
altnmf_data = sys.argv[3]
dcnmf_data = sys.argv[4]
fnmf_data = sys.argv[5]
phi_delta = str(sys.argv[6])
rho = sys.argv[7]
size = sys.argv[8]
seed = sys.argv[9]

data_rows = list(pd.read_csv(sim_data_path +'.rows.csv.gz' )['rows'])
data_cols = list(pd.read_csv(sim_data_path +'.cols.csv.gz' )['cols']) 
result_file = nmf_data_path+'_eval.csv'

phi = phi_delta.split('_')[0]
delta = phi_delta.split('_')[1]

def eval_model(model_file,result_file,mode):

    if mode =='full':
        df_theta = pd.DataFrame(model_file['theta'])
    else:
        df_theta = pd.DataFrame(model_file['corr'])
    
    df_theta.index = data_rows

    df_umap= pd.DataFrame()
    df_umap['cell'] = data_rows

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_theta.to_numpy())
    kmeans = KMeans(n_clusters=25, init='k-means++',random_state=0).fit(scaled)
    df_umap['topic_bulk'] = kmeans.labels_
    df_umap['cell_type'] = [x.split('_')[1] for x in df_umap['cell']]

    for ct in df_umap['cell_type'].unique():
        s = normalized_mutual_info_score(df_umap['cell_type'].values,df_umap['topic_bulk'].values)
    
    # score = normalized_mutual_info_score(df_umap['cell_type'].values,df_umap['topic_bulk'].values)


    result = ['nmi',mode,phi,delta,rho,size,seed,score]
    df_res = pd.DataFrame(result).T
    if os.path.isfile(result_file):
        df_res.to_csv(result_file,index=False, mode='a',header=False)
    else:
        df_res.columns = ['method','mode','phi','delta','rho','size','seed','score']
        df_res.to_csv(result_file,index=False)



    df_match = df_umap[['cell','cell_type','topic_bulk']]
    df_match = df_match.groupby(['cell_type','topic_bulk']).count().reset_index()
    df_match = df_match.pivot('cell_type','topic_bulk')
    df_match = df_match.fillna(0).T
    df_match['assign'] = [x for x in df_match.max(axis=1)]
    score = 1-df_match.apply(lambda x: sum(x[x != x.iloc[-1]]), axis=1).sum()/df_umap.shape[0]


    result = ['purity',mode,phi,delta,rho,size,seed,score]
    df_res = pd.DataFrame(result).T
    if os.path.isfile(result_file):
        df_res.to_csv(result_file,index=False, mode='a',header=False)
    else:
        df_res.columns = ['method','mode','phi','delta','rho','size','seed','score']
        df_res.to_csv(result_file,index=False)


    ### scanpy knn-leiden

    df_theta = pd.DataFrame(model_file['theta'])

    import scanpy as sc
    import anndata

    adata = anndata.AnnData(df_theta.to_numpy())
    dfvars = pd.DataFrame([str(i)+'_t' for i in range(df_theta.shape[1])])
    dfobs = pd.DataFrame(data_rows)
    adata.obs = dfobs
    adata.var = dfvars
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata)
    df_umap['leiden'] = adata.obs['leiden']

    df_match = df_umap[['cell','cell_type','leiden']]
    df_match = df_match.groupby(['cell_type','leiden']).count().reset_index()
    df_match = df_match.pivot('cell_type','leiden')
    df_match = df_match.fillna(0).T
    df_match['assign'] = [x for x in df_match.max(axis=1)]
    score = 1-df_match.apply(lambda x: sum(x[x != x.iloc[-1]]), axis=1).sum()/df_umap.shape[0]


    result = ['leiden',mode,phi,delta,rho,size,seed,score]
    df_res = pd.DataFrame(result).T
    if os.path.isfile(result_file):
        df_res.to_csv(result_file,index=False, mode='a',header=False)
    else:
        df_res.columns = ['method','mode','phi','delta','rho','size','seed','score']
        df_res.to_csv(result_file,index=False)


    # adata = anndata.AnnData(df_theta.to_numpy())
    # dfvars = pd.DataFrame([str(i)+'_t' for i in range(df_theta.shape[1])])
    # dfobs = pd.DataFrame(data_rows)
    # adata.obs = dfobs
    # adata.var = dfvars

    # nbr_dist = get_neighbours_distances(df_theta.to_numpy())
    # spmat = sparse.csr_matrix(nbr_dist)
    # sc.tl.leiden(adata,adjacency=spmat)
    # df_umap['leiden'] = adata.obs['leiden']

    # df_match = df_umap[['cell','cell_type','leiden']]
    # df_match = df_match.groupby(['cell_type','leiden']).count().reset_index()
    # df_match = df_match.pivot('cell_type','leiden')
    # df_match = df_match.fillna(0).T
    # df_match['assign'] = [x for x in df_match.max(axis=1)]
    # score = 1-df_match.apply(lambda x: sum(x[x != x.iloc[-1]]), axis=1).sum()/df_umap.shape[0]


    # result = ['leiden',mode,phi,delta,rho,size,seed,score]
    # df_res = pd.DataFrame(result).T
    # if os.path.isfile(result_file):
    #     df_res.to_csv(result_file,index=False, mode='a',header=False)
    # else:
    #     df_res.columns = ['method','mode','phi','delta','rho','size','seed','score']
    #     df_res.to_csv(result_file,index=False)





alt_model = np.load(altnmf_data)
eval_model(alt_model,result_file,'alt')

dc_model = np.load(dcnmf_data)
eval_model(dc_model,result_file,'dc')

f_model = np.load(fnmf_data)
eval_model(f_model,result_file,'full')




