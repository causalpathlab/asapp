# ######################################################
# ##### bulk setup
# ######################################################

sample = 'gtex_sc'
wdir = 'experiments/asapp/figures/fig_5_a/'


######################################################
##### single cell nmf
######################################################

import asappy
import pandas as pd


# asappy.create_asap_data(sample,working_dirpath=wdir)

n_topics = 25
data_size = 55000
number_batches = 4

asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches,working_dirpath=wdir)
asappy.generate_pseudobulk(asap_object,tree_depth=10)
asappy.asap_nmf(asap_object,num_factors=n_topics,seed=42)
asappy.generate_model(asap_object)



######################################################
##### sc analysis
######################################################

import asappy
import anndata as an
from asappy.plotting.palette import get_colors
from plotnine import *

asap_adata = an.read_h5ad(wdir+'results_main/'+sample+'.h5asapad')

asappy.plot_gene_loading(asap_adata,top_n=10,max_thresh=25)

asappy.leiden_cluster(asap_adata,resolution=1.0)
asappy.run_umap(asap_adata,min_dist=0.5)
asappy.plot_umap(asap_adata,col='cluster',pt_size=0.1)

asap_adata.obs['celltype'] = [ x.split('-')[1].replace('@gtex_sc','') for x in asap_adata.obs.index.values]
asappy.plot_umap(asap_adata,col='celltype',pt_size=0.1)

## get cell type level 2 
# import h5py as hf
# f = hf.File('/data/sishir/data/gtex_sc/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad')
# codes = list(f['obs']['Cell types level 2'])
# cat = [x.decode('utf-8') for x in f['obs']['__categories']['Cell types level 2']]
# f.close()


import h5py as hf

f = hf.File('/data/sishir/data/gtex_sc/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad')
codes = list(f['obs']['Granular cell type'])
cat = [x.decode('utf-8') for x in f['obs']['__categories']['Granular cell type']]
f.close()


catd ={}
for ind,itm in enumerate(cat):catd[ind]=itm
asap_adata.obs['celltype3'] = [ catd[x] for x in codes]
asappy.plot_umap(asap_adata,col='celltype3',pt_size=0.1)
        
asap_adata.write(wdir+'results/'+sample+'.h5asapad')

ftype='pdf'
pt_size=0.1
col='celltype3'

df_umap = pd.DataFrame(asap_adata.obsm['umap_coords'],columns=['umap1','umap2'])
df_umap[col] = pd.Categorical(asap_adata.obs[col].values)
nlabel = asap_adata.obs[col].nunique()
custom_palette = get_colors(nlabel) 

if ftype == 'pdf':
    fname = asap_adata.uns['inpath']+'_'+col+'_'+'umap.pdf'
else:
    fname = asap_adata.uns['inpath']+'_'+col+'_'+'umap.png'

legend_size=7

p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
    geom_point(size=pt_size) +
    scale_color_manual(values=custom_palette)  +
    guides(color=guide_legend(override_aes={'size': legend_size})))

p = p + theme(
    plot_background=element_rect(fill='white'),
    panel_background = element_rect(fill='white'))


p.save(filename = fname, height=8, width=20, units ='in', dpi=600)


fname = asap_adata.uns['inpath']+'_'+col+'_'+'umap.png'
p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
    geom_point(size=pt_size) +
    scale_color_manual(values=custom_palette)  
    )

p = p + theme(
    plot_background=element_rect(fill='white'),
    panel_background = element_rect(fill='white'),
    legend_position= 'none')
p.save(filename = fname, height=8, width=20, units ='in', dpi=600)



col='celltype2'

df_umap = pd.DataFrame(asap_adata.obsm['umap_coords'],columns=['umap1','umap2'])
df_umap[col] = pd.Categorical(asap_adata.obs[col].values)
nlabel = asap_adata.obs[col].nunique()
custom_palette = get_colors(nlabel) 

fname = asap_adata.uns['inpath']+'_'+col+'_'+'umap.pdf'

legend_size=7

p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
    geom_point(size=pt_size) +
    scale_color_manual(values=custom_palette)  +
    guides(color=guide_legend(override_aes={'size': legend_size})))

p = p + theme(
    plot_background=element_rect(fill='white'),
    panel_background = element_rect(fill='white'))


p.save(filename = fname, height=8, width=20, units ='in', dpi=600)


fname = asap_adata.uns['inpath']+'_'+col+'_'+'umap.png'
p = (ggplot(data=df_umap, mapping=aes(x='umap1', y='umap2', color=col)) +
    geom_point(size=pt_size) +
    scale_color_manual(values=custom_palette)  
    )

p = p + theme(
    plot_background=element_rect(fill='white'),
    panel_background = element_rect(fill='white'),
    legend_position= 'none')
p.save(filename = fname, height=8, width=20, units ='in', dpi=600)


###########################
 ## THETA HEATMAP
##########################

sample = 'gtex_sc'
wdir = 'experiments/asapp/figures/fig_5_a/'
asap_adata = an.read_h5ad(wdir+'results/'+sample+'.h5asapad')
#### cells by factor plot 

pmf2t = asappy.pmf2topic(beta=asap_adata.uns['pseudobulk']['pb_beta'] ,theta=asap_adata.obsm['theta'])
df = pd.DataFrame(pmf2t['prop'])
df.columns = ['t'+str(x) for x in df.columns]


df.reset_index(inplace=True)
df['celltype'] = asap_adata.obs['celltype2'].values

dftemp = df['celltype'].value_counts()  
selected_celltype = dftemp[dftemp>1000].index.values

df = df[df['celltype'].isin(selected_celltype)]


def sample_n_rows(group):
    return group.sample(n=min(n, len(group)))

#### theta paper heatmap
n= 3000 ## 1k per cell type for 9 celltypes with total cell >1k
#### total sample of 23732 cells and 25 topics
### total cells 207388 and total cell types 13

sampled_df = df.groupby('celltype', group_keys=False).apply(sample_n_rows)

sampled_df.drop(columns='index',inplace=True)
sampled_df.set_index('celltype',inplace=True)

import matplotlib.pylab as plt
import seaborn as sns

sns.clustermap(sampled_df,cmap='Oranges',col_cluster=False)
plt.savefig(asap_adata.uns['inpath']+'_prop_hmap.png',dpi=600);plt.close()

