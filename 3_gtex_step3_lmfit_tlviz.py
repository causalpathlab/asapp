# ######################################################
# ##### bulk setup
# ######################################################

bk_sample = 'bulk'
sc_sample ='gtex_sc'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/results/mix_'


######################################################
##### transfer learning
######################################################

import asappy
import anndata as an
import pandas as pd
import numpy as np

### get single cell beta, theta, and pbtheta + bulk theta
asap_adata = an.read_h5ad('./results/'+sc_sample+'.h5asapad')
sc_beta = pd.DataFrame(asap_adata.uns['pseudobulk']['pb_beta'].T)
sc_beta.columns = asap_adata.var.index.values


sc_theta = pd.DataFrame(asap_adata.obsm['theta'])
sc_theta.index= asap_adata.obs.index.values


sc_pbtheta = pd.DataFrame(asap_adata.uns['pseudobulk']['pb_theta'])
sc_pbtheta.index = ['pb'+str(x) for x in sc_pbtheta.index.values]
sc_pbtheta.columns = ['t'+str(x) for x in sc_pbtheta.columns]

bulk_theta = pd.read_csv(outpath+'bulk_theta_lmfit.csv.gz')
bulk_theta.index = bulk_theta['Unnamed: 0']
bulk_theta.drop(columns=['Unnamed: 0'],inplace=True)
bulk_theta.columns = ['t'+str(x) for x in bulk_theta.columns]

#### common def
def get_umap(df,md):
	from asappy.clustering import leiden_cluster

	snn,cluster = leiden_cluster(df,resolution=0.1,k=10)

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

############ plot bulk theta only

umap_coords,cluster = get_umap(bulk_theta,0.3)
dfumap = pd.DataFrame(umap_coords[0])
dfumap['cell'] = bulk_theta.index.values
dfumap.columns = ['umap1','umap2','cell']
dfumap['tissue'] = [x.split('@')[1] for x in bulk_theta.index.values]
dfumap['cluster'] = pd.Categorical(cluster)
asappy.plot_umap_df(dfumap,'tissue',outpath+'_bulkonly_')
asappy.plot_umap_df(dfumap,'cluster',outpath+'_bulkonly_')


############################################################
####### plot pseudobulk and bulk together
############################################################



########### normalization ############

sc_pbtheta = sc_pbtheta.div(sc_pbtheta.sum(axis=1), axis=0)
bulk_theta = bulk_theta.div(bulk_theta.sum(axis=1), axis=0)

def quantile_normalize(df):
    df_sorted = pd.DataFrame(np.sort(df.values,
				     axis=0), 
			     index=df.index, 
			     columns=df.columns)
    df_mean = df_sorted.mean(axis=1)
    df_mean.index = np.arange(1, len(df_mean) + 1)
    df_qn =df.rank(method="min").stack().astype(int).map(df_mean).unstack()
    return(df_qn)

combined_df = pd.concat([sc_pbtheta,bulk_theta],axis=0,ignore_index=False)
df = quantile_normalize(combined_df)


################### plot theta separately 
import matplotlib.pylab as plt
import seaborn as sns
sns.clustermap(df.loc[df.index.str.contains('pb'),:]);plt.savefig(outpath+'scpbtheta.png');plt.close()
sns.clustermap(df.loc[~df.index.str.contains('pb'),:]);plt.savefig(outpath+'bulk_theta.png');plt.close()


################# plot theta umap together 
umap_coords,cluster = get_umap(df,0.1)
dfumap = pd.DataFrame(umap_coords[0])
dfumap['cell'] = df.index.values
dfumap.columns = ['umap1','umap2','cell']
dfumap['batch'] = ['pb' if 'pb' in x else 'bulk' for x in dfumap['cell']]
dfumap['cluster'] = cluster
dfumap['cluster'] = pd.Categorical(dfumap['cluster'])
dfumap['tissue'] = ['a' for x in sc_pbtheta.index.values] + [x.split('@')[1] for x in bulk_theta.index.values]


from plotnine  import *
import matplotlib.pylab as plt 


size_mapping = {
'a':10, 
'heart_atrial_appendage':2, 'esophagus_muscularis':2,
'skin_sun_exposed_lower_leg':2, 'muscle_skeletal':2, 'prostate':2,
'skin_not_sun_exposed_suprapubic':2, 'heart_left_ventricle':2,
'breast_mammary_tissue':2, 'lung':2, 'esophagus_mucosa':2
}
shape_mapping = {
'a':'+', 
'heart_atrial_appendage':'o', 'esophagus_muscularis':'o',
'skin_sun_exposed_lower_leg':'o', 'muscle_skeletal':'o', 'prostate':'o',
'skin_not_sun_exposed_suprapubic':'o', 'heart_left_ventricle':'o',
'breast_mammary_tissue':'o', 'lung':'o', 'esophagus_mucosa':'o'
}
custom_palette2 = [
"#d62728",
"#f0f8ff","#f0f8ff","#f0f8ff","#f0f8ff","#f0f8ff",
"#f0f8ff","#f0f8ff","#f0f8ff","#f0f8ff","#f0f8ff"
] 

custom_palette1= [
"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
"#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
"#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
"#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
'#6b6ecf', '#8ca252',  '#8c6d31', '#bd9e39', '#d6616b'
]
def plot_umap_df(dfumap,col,fpath):
	
	nlabel = dfumap[col].nunique() 
	fname = fpath+'_'+col+'_'+'umap.png'

	# pt_size=1.0
	legend_size=7

	p = (ggplot(data=dfumap, mapping=aes(x='umap1', y='umap2', color=col,shape=col,size=col)) +
		geom_point() +
		scale_color_manual(values=custom_palette1)+
  		scale_size_manual(values=size_mapping) + 
  		scale_shape_manual(values=shape_mapping) + 
		guides(color=guide_legend(override_aes={'size': legend_size})))

	p = p + theme(
		plot_background=element_rect(fill='white'),
		panel_background = element_rect(fill='white'))
		
	p.save(filename = fname, height=10, width=15, units ='in', dpi=300)

# plot_umap_df(dfumap,'batch',outpath)
# plot_umap_df(dfumap,'cluster',outpath)
plot_umap_df(dfumap,'tissue',outpath)