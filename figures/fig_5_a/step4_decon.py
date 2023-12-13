
bk_sample = 'bulk'
sc_sample ='gtex_sc'
wdir = 'experiments/asapp/figures/fig_5_a/'
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
k = 250

model_ann = ApproxNN(df.iloc[pb_index,:].to_numpy(),pb_labels)
model_ann.build()
for blk in blk_index:
		cf_idxs = model_ann.query(df.iloc[blk,:].to_numpy(),k)
		nbr.append(cf_idxs)


import h5py as hf

f = hf.File('/data/sishir/data/gtex_sc/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad')
codes = list(f['obs']['Granular cell type'])
cat = [x.decode('utf-8') for x in f['obs']['__categories']['Granular cell type']]
f.close()

catd ={}
for ind,itm in enumerate(cat):catd[ind]=itm


asap_adata.obs['celltype2'] = [ catd[x] for x in codes]

ct_list = asap_adata.obs['celltype2'].values
cts = np.unique(ct_list)


####################
####################
# PB method
#######################
#######################

# pb_map = {}
# pb_counter = 0
# for k in asap_adata.uns['pseudobulk']['pb_map'].keys():
# 	for ik in asap_adata.uns['pseudobulk']['pb_map'][k].keys():
# 		pb_map['pb'+str(pb_counter)]=asap_adata.uns['pseudobulk']['pb_map'][k][ik]
# 		pb_counter += 1


# '''
# for each bulk - get pb nbrs and then get cell type of cells of
# each pb and collect all celltypes
# '''
# bulk_map = {}
# for ni,pbs in enumerate(nbr):
# 	bulk_sample = blk_labels[ni]
# 	bulk_map[bulk_sample] = {}
# 	for pb in pbs:
# 		for ct in ct_list[pb_map[pb]]:
			
# 			if ct not in bulk_map[bulk_sample].keys():
# 				bulk_map[bulk_sample][ct] = 1
# 			else:
# 				bulk_map[bulk_sample][ct] += 1



####################
####################
# SC Method
#######################
#######################


bulk_map = {}
for ni,scs in enumerate(nbr):
	bulk_sample = blk_labels[ni]
	bulk_map[bulk_sample] = {}
	if ni % 100 ==0:
		print(ni)
	for sc in scs:
		ct = catd[codes[np.where(asap_adata.obs.index.values == sc)[0][0]]]
		if ct not in bulk_map[bulk_sample].keys():
			bulk_map[bulk_sample][ct] = 1
		else:
			bulk_map[bulk_sample][ct] += 1



####################
####################
# pb-sc common
#######################
#######################
final_blist = []
for bulk in bulk_map.keys():
    bulk_sample = bulk_map[bulk]
    blist = [bulk]
    for ct in cts:
        if ct in bulk_sample.keys():
            blist.append(bulk_sample[ct])
        else:
            blist.append(0)
    final_blist.append(blist)
    
df = pd.DataFrame(final_blist)
tissue_label = [x.split('@')[1] for x in df[0]]
df.drop(columns=[0],inplace=True)
df.columns = cts
df['tissue'] = tissue_label
dfgrp = df.groupby('tissue').mean()


dftemp = pd.Series(ct_list).value_counts()>1000
outcols = list(dftemp[dftemp==False].index.values)

dfgrp.drop(columns=outcols,inplace=True)

dfn= dfgrp.div(dfgrp.sum(axis=1), axis=0)
dfn = dfn.reset_index()
dfnm = pd.melt(dfn,id_vars='tissue')
dfnm.columns = ['tissue','celltype','prop']

dfnm.to_csv(outpath+'_bulk_deconv_results_theta.csv')


import pandas as pd
from plotnine import *
from asappy.plotting.palette import get_colors
bk_sample = 'bulk'
sc_sample ='gtex_sc'
wdir = 'experiments/asapp/figures/fig_5_a/'
outpath = wdir+'results/'

dfnm = pd.read_csv(outpath+'_bulk_deconv_results_theta.csv')

dfnm = dfnm.sort_values('tissue',ascending=False)

# custom_palette = get_colors(100)
# random.shuffle(custom_palette) 

custom_palette =['#C2FF99',
 '#CB7E98',
 '#CC0744',
 '#3B9700',
 '#FF90C9',
 '#013349',
 '#63FFAC',
 '#404E55',
 '#456D75',
 '#BEC459',
 '#8FB0FF',
 '#372101',
 '#FFAA92',
 '#575329',
 '#B77B68',
 '#6F0062',
 '#A05837',
 '#456648',
 '#FF913F',
 '#549E79',
 '#0AA6D8',
 '#6B7900',
 '#A1C299',
 '#FDE8DC',
 '#A30059',
 '#1E6E00',
 '#C0B9B2',
 '#99ADC0',
 '#A4E804',
 '#997D87',
 '#A3C8C9',
 '#D790FF',
 '#BA0900',
 '#922329',
 '#C2FFED',
 '#201625',
 '#7B4F4B',
 '#00C2A0',
 '#6367A9',
 '#A77500',
 '#B903AA',
 '#D16100',
 '#452C2C',
 '#00846F',
 '#61615A',
 '#0089A3',
 '#FF8A9A',
 '#FF34FF',
 '#001E09',
 '#00FECF',
 '#7A87A1',
 '#00A6AA',
 '#0086ED',
 '#4FC601',
 '#5B4534',
 '#886F4C',
 '#6B002C',
 '#6A3A4C',
 '#B79762',
 '#FFB500',
 '#00489C',
 '#0CBD66',
 '#B4A8BD',
 '#006FA6',
 '#324E72',
 '#1CE6FF',
 '#C8A1A1',
 '#4A3B53',
 '#7A4900',
 '#8CD0FF',
 '#1B4400',
 '#B05B6F',
 '#34362D',
 '#008941',
 '#5A0007',
 '#FFF69F',
 '#FAD09F',
 '#DDEFFF',
 '#7900D7',
 '#FF2F80',
 '#04F757',
 '#A079BF',
 '#3A2465',
 '#788D66',
 '#BC23FF',
 '#0000A6',
 '#885578',
 '#EEC3FF',
 '#72418F',
 '#300018',
 '#636375',
 '#004D43',
 '#FF4A46',
 '#000035',
 '#FFFF00',
 '#938A81',
 '#809693',
 '#9B9700',
 '#3B5DFF',
 '#D157A0',
 '#772600',
 '#FFDBE5']

p = (ggplot(data=dfnm, mapping=aes(x='tissue', y='prop', fill='celltype')) +
	geom_bar(position="stack", stat="identity", size=0) +
	scale_fill_manual(values=custom_palette) +
	coord_flip())


p = p + theme(
	plot_background=element_rect(fill='white'),
	panel_background = element_rect(fill='white'),
	axis_text_x=element_text(rotation=45, hjust=1))
p.save(filename = outpath+'_bulk_deconv_results_theta.pdf', height=10, width=15, units ='in', dpi=600)
# p.save(filename = outpath+'_bulk_deconv_results_theta.png', height=10, width=15, units ='in', dpi=600)





