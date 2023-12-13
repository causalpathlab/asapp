
bk_sample = 'bulk'
sc_sample ='gtex_sc'
wdir = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/figures/fig_5_a/'
outpath = wdir+'results/'

import asappy
import anndata as an
import pandas as pd
import numpy as np



### get single cell asap results and get single cell corr
asap_adata = an.read_h5ad(wdir+'results/'+sc_sample+'.h5asapad')
sc_theta = asap_adata.obsm['theta']
sc_theta = pd.DataFrame(sc_theta)
sc_theta.index =  asap_adata.obs.index.values
sc_theta.columns = ['t'+str(x) for x in sc_theta.columns]


##################################

### get estimated bulk correlation from previous transfer learning step
bulk_corr = pd.read_csv(wdir+'results/mix_bulk_theta_asap.csv.gz')
bulk_corr.index = bulk_corr['Unnamed: 0']
bulk_corr.drop(columns=['Unnamed: 0'],inplace=True)
bulk_corr.columns = ['t'+str(x) for x in bulk_corr.columns]


### combine sc and bulk

df = pd.concat([sc_theta,bulk_corr],axis=0,ignore_index=False)




import numpy as np
import pandas as pd
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


pb_index = [ i for i,x in enumerate(df.index.values) if 'gtex_sc' in x]
pb_labels = df.index.values[pb_index]

blk_index = [ i for i,x in enumerate(df.index.values) if 'gtex_sc' not in x]
blk_labels = df.index.values[blk_index]


nbr = []
k = 100

model_ann = ApproxNN(df.iloc[pb_index,:].to_numpy(),pb_labels)
model_ann.build()
for blk in blk_index:
		cf_idxs = model_ann.query(df.iloc[blk,:].to_numpy(),k)
		nbr.append(cf_idxs)

df_nbr = pd.DataFrame(nbr)
df_nbr.index = blk_labels


selected_sc = np.unique(df_nbr.to_numpy().flatten())
selected_sc_bulk = np.hstack((blk_labels,selected_sc))

df = df[df.index.isin(selected_sc_bulk)]



df_sc = df[df.index.isin(selected_sc)]
df_blk = df[df.index.isin(blk_labels)]



from asappy.util.analysis import quantile_normalization

sc_pbnorm,bulk_norm = quantile_normalization(df_sc.to_numpy(),df_blk.to_numpy())

df_scn = pd.DataFrame(sc_pbnorm)
df_scn.index = df_sc.index.values
df_scn.columns = df_sc.columns

df_blkn = pd.DataFrame(bulk_norm)
df_blkn.index = df_blk.index.values
df_blkn.columns = df_blk.columns

dfn = pd.concat([df_scn,df_blkn],axis=0,ignore_index=False)
dfn = pd.concat([df_sc,df_blk],axis=0,ignore_index=False)



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
umap_coords,cluster = get_umap(dfn.to_numpy(),0.01)
dfumap = pd.DataFrame(umap_coords[0])
dfumap['cell'] = dfn.index.values
dfumap.columns = ['umap1','umap2','cell']

def get_tissue(x):
	if '@gtex_sc' in x: return 'single-cell'
	else : return x.split('@')[1]



dfumap['tissue'] = dfumap['cell'].apply(get_tissue)


# dfumap.to_csv(outpath+'_dfumap_bulk_pb.csv.gz',compression='gzip')
# asappy.plot_umap_df(dfumap,'tissue',outpath)
# asappy.plot_umap_df(dfumap,'batch',outpath)


### for plotting....##

from plotnine  import *
import matplotlib.pylab as plt 
from asappy.plotting.palette import get_colors


# dfumap = pd.read_csv(outpath+'_dfumap_bulk_pb.csv.gz')

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



import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['figure.autolayout'] = True
import colorcet as cc
import seaborn as sns
import networkx as nx

df_umap = pd.read_csv(spr.cell_topic.id +'1_d_celltopic_label.csv.gz')
celltype = 'T/23'
cell = df_umap[df_umap['celltype_ct']==celltype]['cell'].values[100]
df_nbr = spr.cell_topic.neighbour.copy()
nbr_indxs = list(df_nbr[df_nbr['cell']==cell].values[0][1:])
nbr_nodes = list(df_nbr.iloc[nbr_indxs,0].values)

points = [ (x,y) for x,y in zip(df_umap['umap1'],df_umap['umap2'])]
g = nx.Graph()

for node in df_umap['cell']:g.add_node(node)

for nbr in nbr_nodes:g.add_edge(cell,nbr,weight=1)

pos = {node: point for node,point in zip(df_umap['cell'].values,points)}

node_color=[]
node_size =[]
for v in g.nodes():
	if  v in nbr_nodes: 
		node_color.append('limegreen')
		node_size.append(100)
	elif v == cell: 
		print(v)
		node_color.append('orange')
		node_size.append(1000)
	else:
		node_color.append('black')
		node_size.append(0.001)

# edge_color = sns.color_palette("viridis", 150)
nx.draw_networkx_nodes(g,pos=pos,node_size=node_size,node_color=node_color,alpha=0.8)
ax=plt.gca()
for edge in g.edges():
	source, target = edge
	rad = 0.1
	arrowprops=dict(lw=g.edges[(source,target)]['weight'],
					arrowstyle="-",
					color='black',
					connectionstyle=f"arc3,rad={rad}",
					linestyle= '--',
					alpha=0.2)
	ax.annotate("",
				xy=pos[source],
				xytext=pos[target],
				arrowprops=arrowprops
			)
nx.draw_networkx_edges(g,pos=pos,width=0)
plt.savefig(spr.interaction_topic.id+'10_cell_nbr_example.png',dpi=600);plt.close()


