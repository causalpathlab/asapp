# ######################################################
# ##### setup
# ######################################################

bk_sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/mix_'


######################################################
##### transfer learning visualization
######################################################

import asappy
import anndata as an
import pandas as pd
import numpy as np

### get single cell asap results
asap_adata = an.read_h5ad('./results/'+sc_sample+'.h5asapad')


########## estimate single cell correlation using single beta
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
bulk_corr = pd.read_csv(outpath+'bulk_corr_asap.csv.gz')
bulk_corr.index = bulk_corr['Unnamed: 0']
bulk_corr.drop(columns=['Unnamed: 0'],inplace=True)
bulk_corr.columns = ['t'+str(x) for x in bulk_corr.columns]


########### normalization for single cell correlation and bulk correlation ############

from asappy.util.analysis import quantile_normalization

sc_norm,bulk_norm = quantile_normalization(sc_pbcorr.to_numpy(),bulk_corr.to_numpy())

sc_norm = pd.DataFrame(sc_norm)
sc_norm.index = sc_pbcorr.index.values
sc_norm.columns = sc_pbcorr.columns

bulk_norm = pd.DataFrame(bulk_norm)
bulk_norm.index = bulk_corr.index.values
bulk_norm.columns = bulk_corr.columns

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
		data = df.to_numpy(),
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

############ plot bulk correlation only

umap_coords,cluster = get_umap(bulk_norm,0.3)
dfumap = pd.DataFrame(umap_coords[0])
dfumap['cell'] = bulk_corr.index.values
dfumap.columns = ['umap1','umap2','cell']
dfumap['tissue'] = [x.split('@')[1] for x in bulk_corr.index.values]
dfumap['cluster'] = pd.Categorical(cluster)
asappy.plot_umap_df(dfumap,'tissue',outpath+'_bulk_tl_')
asappy.plot_umap_df(dfumap,'cluster',outpath+'_bulk_tl_')


############################################################
####### plot pseudobulk and bulk correlation together
############################################################

df = pd.concat([sc_norm,bulk_norm],axis=0,ignore_index=False)

################# combined umap
umap_coords,cluster = get_umap(df,0.1)
dfumap = pd.DataFrame(umap_coords[0])
dfumap['cell'] = df.index.values
dfumap.columns = ['umap1','umap2','cell']
dfumap['batch'] = ['pb' if 'pb' in x else 'bulk' for x in dfumap['cell']]
dfumap['cluster'] = cluster
dfumap['cluster'] = pd.Categorical(dfumap['cluster'])
dfumap['tissue'] = ['a' for x in sc_pbcorr.index.values] + [x.split('@')[1] for x in bulk_corr.index.values]


from plotnine  import *
import matplotlib.pylab as plt 


size_mapping = {
'a':2, 
'heart_atrial_appendage':1, 'esophagus_muscularis':1,
'skin_sun_exposed_lower_leg':1, 'muscle_skeletal':1, 'prostate':1,
'skin_not_sun_exposed_suprapubic':1, 'heart_left_ventricle':1,
'breast_mammary_tissue':1, 'lung':1, 'esophagus_mucosa':1
}
shape_mapping = {
'a':'+', 
'heart_atrial_appendage':'o', 'esophagus_muscularis':'o',
'skin_sun_exposed_lower_leg':'o', 'muscle_skeletal':'o', 'prostate':'o',
'skin_not_sun_exposed_suprapubic':'o', 'heart_left_ventricle':'o',
'breast_mammary_tissue':'o', 'lung':'o', 'esophagus_mucosa':'o'
}

custom_palette1= [
"#e6194B",  
"#fabed4","#ffd8b1","#fffac8",
"#aaffc3","#dcbeff","#a9a9a9",
"#bfef45","#42d4f4","#469990","#c09999"
]

custom_palette2= [
"#900808",  
"#b8d5e6","#b8d5e6","#b8d5e6",
"#b8d5e6","#b8d5e6","#b8d5e6",
"#b8d5e6","#b8d5e6","#b8d5e6","#b8d5e6"
]

def plot_umap_df(dfumap,col,fpath):
	
	nlabel = dfumap[col].nunique() 
	fname = fpath+'_'+col+'_'+'umap.png'

	# pt_size=1.0
	legend_size=7

	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color=col,shape=col,size=col)) +
		geom_point() +
		scale_color_manual(values=custom_palette2)+
  		scale_size_manual(values=size_mapping) + 
  		scale_shape_manual(values=shape_mapping) + 
		guides(color=guide_legend(override_aes={'size': legend_size})))

	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
		
	p.save(filename = fname, height=10, width=15, units ='in', dpi=300)

plot_umap_df(dfumap,'tissue',outpath)
asappy.plot_umap_df(dfumap,'batch',outpath)
asappy.plot_umap_df(dfumap,'cluster',outpath)


################# combined proportion heat map between bulk correction and sc correlation

dfh = dfumap[['cell','cluster','tissue']]
dfh = dfh.groupby(['cluster','tissue'])['cell'].count().reset_index().sort_values(['cluster','cell'])
dfh = dfh.pivot(index='tissue',columns='cluster',values='cell')
dfh =  dfh.div(dfh.sum(0))
dfh.rename(index={'a':'pseudo-bulk'},inplace=True)

import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6)) 

sns.heatmap(dfh,cmap='viridis')
plt.tight_layout()
plt.savefig(outpath+'_prop_hmap.png');plt.close()



################### plot theta for single cell pseudo bulk and bulk separately 
import matplotlib.pylab as plt
import seaborn as sns

pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.uns['pseudobulk']['pb_theta'])
dfpb = pd.DataFrame(pmf2t['prop'])
sns.clustermap(dfpb,cmap='viridis');plt.savefig(outpath+'sc_pbtheta_hmap.png');plt.close()


pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=pred.theta)
dfbulk = pd.DataFrame(pmf2t['prop'])
sns.clustermap(dfbulk,cmap='viridis');plt.savefig(outpath+'bulk_predtheta_hmap.png');plt.close()

