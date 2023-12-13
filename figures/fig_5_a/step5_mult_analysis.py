# ######################################################
# ##### bulk setup
# ######################################################

import asappy
import pandas as pd
import anndata as an
import numpy as np
import asapc



samples = ['gtex_sc_1','gtex_sc_2','gtex_sc_3','gtex_sc_4']
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_5_a/'

for seed,sample in enumerate(samples):

    n_topics = 25
    data_size = 55000
    number_batches = 4

    asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)
    asappy.generate_pseudobulk(asap_object,tree_depth=10)
    asappy.asap_nmf(asap_object,num_factors=n_topics,seed=seed)
    asappy.generate_model(asap_object)


bk_sample = 'bulk'
outpath = wdir+'results/'


for sc_sample in samples:


    ######################################################
    ##### transfer learning
    ######################################################


    ### read single cell nmf
    asap_adata = an.read_h5ad(wdir+'results/'+sc_sample+'.h5asap')
    sc_beta = pd.DataFrame(asap_adata.uns['pseudobulk']['pb_beta'].T)
    sc_beta.columns = asap_adata.var.index.values

    ## read bulk raw data 
    dfbulk = pd.read_csv(outpath+'bulk.csv.gz')
    dfbulk.set_index('Unnamed: 0',inplace=True)
    dfbulk = dfbulk.astype(float)
    bulk_sample_ids = dfbulk.index.values


    ## select common genes 
    common_genes = [ x for x in sc_beta.columns if x in dfbulk.columns.values]
    dfbulk = dfbulk.loc[:,common_genes]

    sc_beta = sc_beta.loc[:,common_genes]


    ## normalize bulk
    row_sums = dfbulk.sum(axis=1)
    target_sum = np.median(row_sums)
    row_sums = row_sums/target_sum
    dfbulk = dfbulk.to_numpy()/row_sums[:, np.newaxis]
        
    #### estimate correlation and theta using single cell beta

    beta_log_scaled = asap_adata.uns['pseudobulk']['pb_beta_log_scaled'] 
    pred_model = asapc.ASAPaltNMFPredict(dfbulk.T,beta_log_scaled)
    pred = pred_model.predict()

    bulk_corr = pd.DataFrame(pred.corr)
    bulk_corr.index = bulk_sample_ids
    bulk_theta = pd.DataFrame(pred.theta)
    bulk_theta.index = bulk_sample_ids
    bulk_corr.to_csv(outpath+'mix_bulk_corr_asap_'+sc_sample+'_.csv.gz',compression='gzip')
    bulk_theta.to_csv(outpath+'mix_bulk_theta_asap_'+sc_sample+'_.csv.gz',compression='gzip')




bk_sample = 'bulk'
sc_sample ='gtex_sc'
outpath = wdir+'results/'


######################################################
##### transfer learning visualization
######################################################


df_pb = pd.DataFrame()
df_bulk = pd.DataFrame()

for sc_sample in samples:
        
    asap_adata = an.read_h5ad(wdir+'results/'+sc_sample+'.h5asap')
    beta_log_scaled = asap_adata.uns['pseudobulk']['pb_beta_log_scaled'] 
    pb_data = asap_adata.uns['pseudobulk']['pb_data'] 

    pred_model = asapc.ASAPaltNMFPredict(pb_data,beta_log_scaled)
    pred = pred_model.predict()

    sc_pbcorr = pd.DataFrame(pred.corr)
    sc_pbcorr.index = ['pb'+str(x) for x in sc_pbcorr.index.values]
    sc_pbcorr.columns = ['t'+str(x) for x in sc_pbcorr.columns]

    df_pb = pd.concat([df_pb,sc_pbcorr],ignore_index=False,axis=0)



    ### get estimated bulk correlation from previous transfer learning step
    bulk_corr = pd.read_csv(wdir+'results/mix_bulk_corr_asap_'+sc_sample+'_.csv.gz')
    bulk_corr.index = bulk_corr['Unnamed: 0']
    bulk_corr.drop(columns=['Unnamed: 0'],inplace=True)
    bulk_corr.columns = ['t'+str(x) for x in bulk_corr.columns]

    df_bulk = pd.concat([df_bulk,bulk_corr],ignore_index=False,axis=0)


    ########### normalization for single cell pseudobulk correlation and bulk correlation ############

from asappy.util.analysis import quantile_normalization

sc_pbnorm,bulk_norm = quantile_normalization(df_pb.to_numpy(),df_bulk.to_numpy())

sc_pbnorm = pd.DataFrame(sc_pbnorm)
sc_pbnorm.index = df_pb.index.values
sc_pbnorm.columns = df_pb.columns

bulk_norm = pd.DataFrame(bulk_norm)
bulk_norm.index = df_bulk.index.values
bulk_norm.columns = df_bulk.columns


df = pd.concat([sc_pbnorm,bulk_norm],axis=0,ignore_index=False)
    
df.to_csv(outpath+'_bulk_pbulk_norm_data.csv.gz',compression='gzip') 


bk_sample = 'bulk'
sc_sample ='gtex_sc'
outpath = wdir+'results/'


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

df = pd.read_csv(outpath+'_bulk_pbulk_norm_data.csv.gz')
df.set_index('Unnamed: 0',inplace=True)



################# combined umap
umap_coords,cluster = get_umap(df.to_numpy(),0.3)
dfumap = pd.DataFrame(umap_coords[0])
# dfumap['cell'] = [x.split('#')[0] for x in df.index.values]
dfumap['cell'] = df.index.values
dfumap.columns = ['umap1','umap2','cell']


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


### good until here....##

from plotnine  import *
import matplotlib.pylab as plt 


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

custom_palette = [
"#aec7e8", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
"#8c564b", "#e377c2", "#1f77b4", "#bcbd22", "#17becf",
 "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
"#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
'#6b6ecf', '#8ca252',  '#8c6d31', '#bd9e39', '#d6616b'
]
def plot_umap_df(dfumap,col,fpath):
	
	nlabel = dfumap[col].nunique() 
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

import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6)) 

sns.heatmap(dfh,cmap='viridis')
plt.tight_layout()
plt.savefig(outpath+'_prop_hmap.png',dpi=600);plt.close()

