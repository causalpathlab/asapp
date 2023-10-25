# ######################################################
# ##### bulk setup
# ######################################################

sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/'+sample
# ######################################################
# ##### bulk data merge
# ######################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py as hf
import glob

datasets = glob.glob('node_data/gtex_bulk/raw/'+'*.gct.gz')


for i,ds in enumerate(datasets):
        df = pd.read_csv(ds,sep='\t',skiprows=2)
        df = df[~df['Description'].str.contains('MT-')]
        df = df[~df['Description'].str.contains('_RNA')]
        df = df[~df['Description'].str.contains('_rRNA')]
        df = df.drop(columns=['id','Description'])
        df = df.T
        df.columns = df.iloc[0,:]
        df = df.iloc[1:,:]
        df.columns = [x.split('.')[0] for x in df.columns]
        tissue = ds.replace('node_data/gtex_bulk/raw/gene_reads_2017-06-05_v8_','').split('.')[0]
        df.index = [x+'@'+tissue for x  in df.index.values]
        df = df[df.columns.sort_values()]
        if i ==0:
                dfmain = df
        else:
                dfmain = pd.concat([dfmain,df])
                print(dfmain.shape)




f = hf.File('data/gtex_sc.h5','r')  
sc_genes = [x.decode('utf-8') for x in f['matrix']['features']['id']]
f.close()

common_genes = [x for x in sc_genes if x in dfmain.columns]
common_genes = list(set(common_genes))
common_genes.sort()
dfmain = dfmain[common_genes]
dfmain = dfmain.loc[:, ~dfmain.columns.duplicated()]
pd.DataFrame(common_genes).to_csv(outpath+'_sc_common_genes.csv',index=False)

dfmain.to_csv(outpath+'.csv.gz',compression='gzip')


# ######################################################
# ##### bulk nmf
# ######################################################

# dfmain = pd.read_csv(outpath+'.csv.gz')
mtx = dfmain.to_numpy().T
num_factors = 10
import asapc
nmf_model = asapc.ASAPdcNMF(mtx,num_factors)
nmf = nmf_model.nmf()

scaler = StandardScaler()
beta_log_scaled = scaler.fit_transform(nmf.beta_log)


reg_model = asapc.ASAPaltNMFPredict(mtx,beta_log_scaled)
reg = reg_model.predict()

np.savez(outpath+'_nmf',
        beta = nmf.beta,
        theta = reg.theta,
        corr = reg.corr)		


# ######################################################
# ##### bulk analysis
# ######################################################

import asappy
import anndata as an
import numpy as np
import pandas as pd

model = np.load(outpath+'_nmf.npz',allow_pickle=True)
asapadata = an.AnnData(shape=(model['theta'].shape[0],model['beta'].shape[0]))
asapadata.obs_names = [ 'b'+str(x) for x in range(model['theta'].shape[0])]
asapadata.var_names = pd.read_csv('./results/bulkcommon_genes.csv').values.flatten()
asapadata.varm['beta'] = model['beta']
asapadata.obsm['theta'] = model['theta']
asapadata.obsm['corr'] = model['corr']
asapadata.uns['inpath'] = outpath

asappy.leiden_cluster(asapadata,k=10,mode='corr',resolution=0.1)
asappy.run_umap(asapadata,min_dist=0.5)
asappy.plot_umap(asapadata,col='cluster')
asappy.plot_gene_loading(asapadata)

dfmain = pd.read_csv(outpath+'.csv.gz')
dfmain.set_index('Unnamed: 0',inplace=True)
asapadata.obs['celltype'] = [x.split('@')[1] for x in dfmain.index.values]
asappy.plot_umap(asapadata,col='celltype')

