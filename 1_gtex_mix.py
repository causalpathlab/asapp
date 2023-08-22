import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler,scale
import h5py as hf
from asap.util import analysis

import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

out = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/gtex_mix/'
bulk_model = np.load('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/gtex_bulk/bulk_dcnmf.npz',allow_pickle=True)
sc_model = np.load('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/pbmc/pbmc_dcnmf.npz',allow_pickle=True)

bulk_df_beta = pd.DataFrame(bulk_model['nmf_beta'].T)
bulk_df_beta.columns = pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/gtex_bulk/bulk_filt_genes.csv')['Description'].values

bulk_df_top = analysis.get_topic_top_genes(bulk_df_beta.iloc[:,:],top_n=10)
bulk_df_top = bulk_df_top.pivot(index='Topic',columns='Gene',values='Proportion')
sns.clustermap(bulk_df_top.T,cmap='viridis')
plt.savefig(out+'bulk_beta1.png');plt.close()


sc_df_beta = pd.DataFrame(sc_model['nmf_beta'].T)
scf = hf.File('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/data/gtex_sc/gtex_sc.h5')
sc_df_beta.columns = [x.decode('utf-8') for x in scf['gtex_sc']['genes'] ]
sc_df_top = analysis.get_topic_top_genes(sc_df_beta.iloc[:,:],top_n=10)
sc_df_top = sc_df_top.pivot(index='Topic',columns='Gene',values='Proportion')
sns.clustermap(sc_df_top.T,cmap='viridis')
plt.savefig(out+'sc_beta1.png');plt.close()


bulk_df_corr = pd.DataFrame(bulk_model['predict_corr'])
bulk_df_corr = scale(bulk_df_corr.to_numpy(),axis=1) * 1e6
sc_df_corr = pd.DataFrame(sc_model['predict_corr'])


import random 
N = 578
minib_i = random.sample(range(0,sc_df_corr.shape[0]),N)
sc_df_corr = sc_df_corr.iloc[minib_i,:]

sc_df_corr = scale(sc_df_corr.to_numpy(),axis=1) * 1e6
df_corr = np.vstack((bulk_df_corr,sc_df_corr))
df_corr = pd.DataFrame(df_corr)


batch_label = (['bulk' if x<578 else 'sc' for x in df_corr.index.values])


from asap.util import batch_correction as bc 


df_corr = pd.DataFrame(bc.batch_correction_scanorama(df_corr.to_numpy(),np.array(batch_label),alpha=0.001,sigma=15))
df_corr.index = df_corr.index


## assign ids
df_umap= pd.DataFrame()
df_umap['cell'] =[x for x in df_corr.index.values]

## assign batch
batch_label = (['bulk' if x<459 else 'sc' for x in df_corr.index.values])
df_umap['batch'] = batch_label

## assign topic
num_factors = 10
kmeans = KMeans(n_clusters=num_factors, init='k-means++',random_state=0).fit(df_corr)
df_umap['asap_topic'] = kmeans.labels_

df_umap[['umap_1','umap_2']] = analysis.get2dprojection(df_corr.to_numpy(),mindist=0.4)
analysis.plot_umaps(df_umap,out+'_batchcorrection.png')



# df_umap_sc = df_umap[['asap_topic','batch']]
# df_umap_sc[['umap_1','umap_2']] = analysis.get2dprojection(df_corr_sc.to_numpy(),mindist=0.4)

# analysis.plot_umaps(df_umap_sc,out+'_post_batchcorrection.png')

for selected in ['bulk']:
    df_umap['selected'] = [x if selected in x else 'sc' for x in df_umap['batch']]
    analysis.plot_umaps(df_umap,out+'_post_batchcorrection_'+'_'+selected+'.png','selected')
    # df_umap_sc['selected'] = [x if selected in x else 'others' for x in df_umap_sc['celltype']]
    # analysis.plot_umaps(df_umap_sc,dl.outpath+'_post_batchcorrection_'+'_'+selected+'2_hm.png','selected')


