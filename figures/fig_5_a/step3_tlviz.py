# ######################################################
# ##### setup
# ######################################################

bk_sample = 'bulk'
sc_sample ='gtex_sc'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_5_a/'
outpath = wdir+'results/'


######################################################
##### transfer learning visualization
######################################################

import asappy
import anndata as an
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

### get single cell asap results
asap_adata = an.read_h5ad(wdir+'results/'+sc_sample+'.h5asapad')


###### pseudobulk
########## estimate single cell pseudobulk correlation using single beta
import asapc

beta_log_scaled = asap_adata.uns['pseudobulk']['pb_beta_log_scaled'] 
pb_data = asap_adata.uns['pseudobulk']['pb_data'] 

pred_model = asapc.ASAPaltNMFPredict(pb_data,beta_log_scaled)
pred = pred_model.predict()

sc_pbcorr = pd.DataFrame(pred.corr)
sc_pbcorr.index = ['pb'+str(x) for x in sc_pbcorr.index.values]
sc_pbcorr.columns = ['t'+str(x) for x in sc_pbcorr.columns]


##################################

### get estimated bulk correlation from previous transfer learning step
bulk_corr = pd.read_csv(wdir+'results/mix_bulk_corr_asap.csv.gz')
bulk_corr.index = bulk_corr['Unnamed: 0']
bulk_corr.drop(columns=['Unnamed: 0'],inplace=True)
bulk_corr.columns = ['t'+str(x) for x in bulk_corr.columns]


########### normalization for single cell pseudobulk correlation and bulk correlation ############

from asappy.util.analysis import quantile_normalization

sc_pbnorm,bulk_norm = quantile_normalization(sc_pbcorr.to_numpy(),bulk_corr.to_numpy())

sc_pbnorm = pd.DataFrame(sc_pbnorm)
sc_pbnorm.index = sc_pbcorr.index.values
sc_pbnorm.columns = sc_pbcorr.columns

bulk_norm = pd.DataFrame(bulk_norm)
bulk_norm.index = bulk_corr.index.values
bulk_norm.columns = bulk_corr.columns


# umap_coords,cluster = get_umap(bulk_norm,0.3)
# dfumap = pd.DataFrame(umap_coords[0])
# dfumap['cell'] = bulk_corr.index.values
# dfumap.columns = ['umap1','umap2','cell']
# dfumap['tissue'] = [x.split('@')[1] for x in bulk_corr.index.values]
# dfumap['cluster'] = pd.Categorical(cluster)
# asappy.plot_umap_df(dfumap,'tissue',outpath+'_bulk_only_tl_')
# asappy.plot_umap_df(dfumap,'cluster',outpath+'_bulk_only_tl_')

'''
############################################################
####### plot pseudobulk, single cell and bulk correlation together
############################################################

# sc_corr = pd.DataFrame(asap_adata.obsm['corr'])
# sc_corr.index = asap_adata.obs.index.values
# sc_corr.columns = ['t'+str(x) for x in sc_corr.columns]

# ### sample 600 sc cells from each celltype such that total sc is about 5k
# sample_size= 500
# sampled_data = pd.DataFrame()
# grouped = asap_adata.obs.groupby('celltype2')

# for group_name, group_data in grouped:
#     if len(group_data) >= sample_size:
#         sampled_group = group_data.sample(n=sample_size, random_state=42)  
#         sampled_data = sampled_data.append(sampled_group)
#     else:
#         sampled_data = sampled_data.append(group_data)

# # Get the index values of the sampled data
# sample_indxs = sampled_data.index.values
# sample_celltype = sampled_data.celltype2.values

# # sample_indxs = asap_adata.obs.groupby(['celltype2']).sample(n).index.values
# sc_corr = sc_corr.loc[sampled_data.index.values,:]
# sc_corr.index = [x.replace('gtex_sc','sc_'+y)for x,y in zip(sc_corr.index.values,sample_celltype)]

# now we have three dataframes
# sc_corr - main single cell 200k data with prediction for corr
# sc_pbcorr - pseudobulk from main single cell 200k data with prediction for corr
# bulk_corr - transfer learning from single cell asap

# bulk_sc_corr = pd.concat([bulk_corr,sc_corr],axis=0,ignore_index=False)

# sc_pbnorm,bulk_sc_norm = quantile_normalization(sc_pbcorr.to_numpy(),bulk_sc_corr.to_numpy())

# sc_pbnorm = pd.DataFrame(sc_pbnorm)
# sc_pbnorm.index = sc_pbcorr.index.values
# sc_pbnorm.columns = sc_pbcorr.columns

# bulk_sc_norm = pd.DataFrame(bulk_sc_norm)
# bulk_sc_norm.index = bulk_sc_corr.index.values
# bulk_sc_norm.columns = ['t'+str(x) for x in bulk_sc_norm.columns]
'''

df = pd.concat([sc_pbnorm,bulk_norm],axis=0,ignore_index=False)
	
df.to_csv(outpath+'_bulk_pbulk_norm_data.csv.gz',compression='gzip') 
####################################


bk_sample = 'bulk'
sc_sample ='gtex_sc'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_5_a/'
outpath = wdir+'results/'

import asappy
import anndata as an
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

### get single cell asap results
asap_adata = an.read_h5ad(wdir+'results/'+sc_sample+'.h5asapad')
df = pd.read_csv(outpath+'_bulk_pbulk_norm_data.csv.gz')
df.set_index('Unnamed: 0',inplace=True)

######################

def get_umap(df,md):
	from asappy.clustering import leiden_cluster

	snn,cluster = leiden_cluster(df,resolution=0.5)

	from umap.umap_ import find_ab_params, simplicial_set_embedding

	min_dist = md
	n_components = 2
	spread: float = 1.0
	alpha: float = 1.0
	gamma: float = 1.0
	negative_sample_rate: int = 5
	maxiter = None
	default_epochs = 500 if snn.shape[0] <= 10000 else 200
	n_epochs = default_epochs if maxiter is None else maxiter
	random_state = np.random.RandomState(42)

	a, b = find_ab_params(spread, min_dist)

	umap_coords = simplicial_set_embedding(
		data = df,
		graph = snn,
		n_components=n_components,
		initial_alpha = alpha,
		a = a,
		b = b,
		gamma = gamma,
		negative_sample_rate = negative_sample_rate,
		n_epochs = n_epochs,
		init='spectral',
		random_state = random_state,
		metric = 'cosine',
		metric_kwds = {},
		densmap=False,
		densmap_kwds={},
		output_dens=False
		)
	return umap_coords,cluster

############ if need to plot bulk correlation only

################# combined umap
umap_coords,cluster = get_umap(df.to_numpy(),0.3)
dfumap = pd.DataFrame(umap_coords[0])
dfumap['cell'] = df.index.values
dfumap.columns = ['umap1','umap2','cell']

def get_batch(x):
    if 'pb' in x: return '0_pb'
    elif 'sc_' in x: return x.split('@')[1]
    else : return '1_bulk'

def get_tissue(x):
	if 'pb' in x: return '0_pb'
	elif 'sc_' in x: return '0_single-cell'
	else : return x.split('@')[1]



# dfumap['batch'] = dfumap['cell'].apply(get_batch)
dfumap['cluster'] = cluster
dfumap['cluster'] = pd.Categorical(dfumap['cluster'])
dfumap['tissue'] = dfumap['cell'].apply(get_tissue)


dfumap.to_csv(outpath+'_dfumap_bulk_pb.csv.gz',compression='gzip')
# asappy.plot_umap_df(dfumap,'tissue',outpath)
# asappy.plot_umap_df(dfumap,'batch',outpath)


### for plotting....##

from plotnine  import *
import matplotlib.pylab as plt 
from asappy.plotting.palette import get_colors


dfumap = pd.read_csv(outpath+'_dfumap_bulk_pb.csv.gz')

size_mapping = {
'sc_Epithelial cell':1,
'sc_Muscle':1,
'sc_Fibroblast':1,
'sc_Endothelial cell':1,
'sc_Immune (myeloid)':1,
'sc_Immune (lymphocyte)':1,
'sc_Adipocyte':1,
'sc_Epithelial cell (keratinocyte)':1,
'sc_Glia':1,
'sc_Stromal':1,
'sc_Other':1,
'sc_Melanocyte':1,
'sc_Neuron':1,
'0_pb':3, 
'heart_atrial_appendage':1, 'esophagus_muscularis':1,
'skin_sun_exposed_lower_leg':1, 'muscle_skeletal':1, 'prostate':1,
'skin_not_sun_exposed_suprapubic':1, 'heart_left_ventricle':1,
'breast_mammary_tissue':1, 'lung':1, 'esophagus_mucosa':1
}

shape_mapping = {
'sc_Epithelial cell':'o',
'sc_Muscle':'o',
'sc_Fibroblast':'o',
'sc_Endothelial cell':'o',
'sc_Immune (myeloid)':'o',
'sc_Immune (lymphocyte)':'o',
'sc_Adipocyte':'o',
'sc_Epithelial cell (keratinocyte)':'o',
'sc_Glia':'o',
'sc_Stromal':'o',
'sc_Other':'o',
'sc_Melanocyte':'o',
'sc_Neuron':'o',
'0_pb':'+', 
'heart_atrial_appendage':'8', 'esophagus_muscularis':'8',
'skin_sun_exposed_lower_leg':'8', 'muscle_skeletal':'8', 'prostate':'8',
'skin_not_sun_exposed_suprapubic':'8', 'heart_left_ventricle':'8',
'breast_mammary_tissue':'8', 'lung':'8', 'esophagus_mucosa':'8'
}



def plot_umap_df(dfumap,col,fpath):
	custom_palette = custom_palette = get_colors(dfumap[col].nunique())

	fname = fpath+'_'+col+'_'+'umap.pdf'

	# pt_size=1.0
	legend_size=7

	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color=col,shape=col,size=col)) +
		geom_point() +
		scale_color_manual(values=custom_palette)+
  		scale_size_manual(values=size_mapping) + 
  		scale_shape_manual(values=shape_mapping) + 
		guides(color=guide_legend(override_aes={'size': legend_size})))

	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
		
	p.save(filename = fname, height=10, width=15, units ='in', dpi=600)


plot_umap_df(dfumap,'tissue',outpath)


################# combined proportion heat map between bulk correction and sc correlation

dfh = dfumap[['cell','cluster','tissue']]
dfh = dfh.groupby(['cluster','tissue'])['cell'].count().reset_index().sort_values(['cluster','cell'])
dfh = dfh.pivot(index='tissue',columns='cluster',values='cell')
dfh =  dfh.div(dfh.sum(0))
dfh.rename(index={'a':'pseudo-bulk'},inplace=True)

dfh.fillna(0,inplace=True)
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6)) 

sns.heatmap(dfh,cmap='Blues')
plt.tight_layout()
plt.savefig(outpath+'_prop_hmap.png',dpi=600);plt.close()

