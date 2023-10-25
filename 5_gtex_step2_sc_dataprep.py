import pandas as pd
import numpy as np
import h5py as hf

from scipy.sparse import csr_matrix

##################
## generate GTEX data in asap input format
##################


# for tissue in ['Breast',
#  'Esophagus mucosa',
#  'Esophagus muscularis',
#  'Heart',
#  'Lung',
#  'Prostate',
#  'Skeletal muscle',
#  'Skin']:
    
f = hf.File('node_data/gtex_sc/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad','r')  
pd.Series(f['obs']['Tissue']).value_counts().sort_index() 
list(f['obs']['__categories']['Tissue']) 

mtx_indptr = f['X']['indptr']
mtx_indices = f['X']['indices']
mtx_data = f['X']['data']

num_rows = len(f['obs']['_index'])
num_cols = len(f['var']['_index'])



rows = csr_matrix((mtx_data,mtx_indices,mtx_indptr),shape=(num_rows,num_cols))
mtx= rows.todense()

## get tissue specific data
# codes = list(f['obs']['Tissue'])
# cat = [x.decode('utf-8') for x in f['obs']['__categories']['Tissue']]
# catd ={}
# for ind,itm in enumerate(cat):catd[ind]=itm
# tissue = [catd[x] for x in codes]


smat = csr_matrix(mtx)

genes = f['var']['gene_ids']

mainf = f
sample='node_data/gtex_sc/gtex_sc'
f = hf.File(sample+'.h5','w')


grp = f.create_group('matrix')
grp.create_dataset('barcodes',data=mainf['obs']['_index'])
grp.create_dataset('indptr',data=smat.indptr)
grp.create_dataset('indices',data=smat.indices)
grp.create_dataset('data',data=smat.data,dtype=np.int32)

g1 = grp.create_group('features')
g1.create_dataset('id',data=genes)

grp.create_dataset('shape',data=mtx.shape)
f.close()


##################
