import sys
sys.path.append('../')
from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.util import analysis
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import h5py as hf

plt.figure(figsize=(10,6))
plt.tight_layout()

experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id


tree_max_depth = 10
num_factors = 10
batch_size = 25000
downsample_pseudobulk = True
downsample_size = 100

dl = DataSet(sample_in,sample_out)
sample_list = dl.get_dataset_names()
dl.initialize_data(sample_list,batch_size)

print(dl.inpath)
print(dl.outpath)


# 
model = np.load(sample_out+'_dcnmf.npz',allow_pickle=True)

# 
df_beta = pd.DataFrame(model['nmf_beta'].T)
df_beta.columns = dl.genes
df_top = analysis.get_topic_top_genes(df_beta.iloc[:,:],top_n=10)
df_top = df_top.pivot(index='Topic',columns='Gene',values='Proportion')
# df_top[df_top>20] = 20
sns.clustermap(df_top.T,cmap='viridis')
plt.savefig(dl.outpath+'beta.png');plt.close()


scaler = StandardScaler()
df_corr = pd.DataFrame(scaler.fit_transform(model['predict_corr']))
# df_corr = pd.DataFrame(model['predict_corr'])
df_corr.index = model['predict_barcodes']
df_corr.index = dl.barcodes

mtx = df_corr.to_numpy()
batch_label = ([x.split('@')[1] for x in df_corr.index.values])


f = hf.File(dl.inpath+'.h5','r')
ct = ['ct'+str(x) for x in f['pbmc']['cell_type']]
f.close()


###########

df_umap= pd.DataFrame()
df_umap['cell'] = df_corr.index.values

kmeans = KMeans(n_clusters=10, init='k-means++',random_state=0).fit(df_corr)
df_umap['topic_bulk'] = kmeans.labels_
df_umap['batch'] = batch_label

############### bc

from asap.util import batch_correction as bc 

###############  bbknn 

adata = bc.batch_correction_bbknn(mtx,batch_label,df_corr.index.values,dl.genes)
df_umap['umap1'] = adata.obsm['X_umap'][:,0]
df_umap['umap2'] = adata.obsm['X_umap'][:,1]


cp = sns.color_palette(cc.glasbey_dark, n_colors=len(df_umap['topic_bulk'].unique()))
p = sns.scatterplot(data=df_umap, x='umap1', y='umap2', hue='topic_bulk',s=5,palette=cp,legend=True)
plt.legend(title='Topic',title_fontsize=18, fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
# p.axes.set_title("topics from bulkNMF",fontsize=30)
p.set_xlabel("UMAP1",fontsize=20)
p.set_ylabel("UMAP2",fontsize=20)
plt.savefig(dl.outpath+'theta_topic_bbknn.png', bbox_inches='tight');plt.close()

cp = sns.color_palette(cc.glasbey_dark, n_colors=len(df_umap['batch'].unique()))
p = sns.scatterplot(data=df_umap, x='umap1', y='umap2', hue='batch',s=5,palette=cp,legend=True)
plt.legend(title='batch',title_fontsize=18, fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(dl.outpath+'theta_batch_bbknn.png', bbox_inches='tight');plt.close()

df_umap['ct'] = ct
cp = sns.color_palette(cc.glasbey_dark, n_colors=len(df_umap['ct'].unique()))
p = sns.scatterplot(data=df_umap, x='umap1', y='umap2', hue='ct',s=5,palette=cp,legend=True)
plt.legend(title='ct',title_fontsize=18, fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(dl.outpath+'theta_batch_ct_bbknn.png', bbox_inches='tight');plt.close()

############### scanorama 

df_corr_sc = pd.DataFrame(bc.batch_correction_scanorama(mtx,batch_label,alpha=0.01,sigma=15))
# df_corr_sc.index = model['predict_barcodes']
df_corr_sc.index = dl.barcodes
np.corrcoef(df_corr.iloc[:,0],df_corr_sc.iloc[:,0])

df_umap_sc = df_umap[['cell','topic_bulk','batch','ct']]

umap_2d = umap.UMAP(n_components=2, init='random', random_state=0,min_dist=0.4,metric='cosine')
proj_2d = umap_2d.fit(df_corr_sc)
df_umap_sc[['umap1','umap2']] = umap_2d.embedding_[:,[0,1]]

cp = sns.color_palette(cc.glasbey_dark, n_colors=len(df_umap_sc['topic_bulk'].unique()))
p = sns.scatterplot(data=df_umap_sc, x='umap1', y='umap2', hue='topic_bulk',s=5,palette=cp,legend=True)
plt.legend(title='Topic',title_fontsize=18, fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
# p.axes.set_title("topics from bulkNMF",fontsize=30)
p.set_xlabel("UMAP1",fontsize=20)
p.set_ylabel("UMAP2",fontsize=20)
plt.savefig(dl.outpath+'theta_topic_sc.png', bbox_inches='tight');plt.close()

df_umap_sc['batch'] = [x.split('-')[0]for x in df_umap_sc['cell']]
cp = sns.color_palette(cc.glasbey_dark, n_colors=len(df_umap_sc['batch'].unique()))
p = sns.scatterplot(data=df_umap_sc, x='umap1', y='umap2', hue='batch',s=5,palette=cp,legend=True)
plt.legend(title='batch',title_fontsize=18, fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(dl.outpath+'theta_batch_sc.png', bbox_inches='tight');plt.close()

cp = sns.color_palette(cc.glasbey_dark, n_colors=len(df_umap_sc['ct'].unique()))
p = sns.scatterplot(data=df_umap_sc, x='umap1', y='umap2', hue='ct',s=5,palette=cp,legend=True)
plt.legend(title='ct',title_fontsize=18, fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(dl.outpath+'theta_batch_ct_sc.png', bbox_inches='tight');plt.close()

#############

