######################################################
##### asap step 2 - pseudobulk analysis
######################################################

sample='tabula_sapiens'
import asappy

data_size = 23000
number_batches = 10
K = 50

asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

for ds in asap_object.adata.uns['dataset_list']:
    asap_object.adata.uns['dataset_batch_size'][ds] = 1000

asappy.generate_pseudobulk(asap_object,tree_depth=10)
asappy.asap_nmf(asap_object,num_factors=K)
asappy.save_model(asap_object)


#### nmf analysis
import anndata as an

asap_adata = an.read_h5ad('./results/'+sample+'.h5asap')

asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=10000)
asappy.plot_gene_loading(asap_adata,top_n=5,max_thresh=200)


asappy.leiden_cluster(asap_adata,k=10,mode='corr',resolution=0.1)
asappy.run_umap(asap_adata,distance='cosine',min_dist=0.1)
asappy.plot_umap(asap_adata,col='cluster')

asap_adata.write('./results/'+sample+'.h5asapad')
asap_adata = an.read_h5ad('./results/'+sample+'.h5asapad')

# asap_adata.obs.cluster.value_counts()



## assign celltype
import h5py as hf
import pandas as pd
inpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/node_data/tabula_sapiens/'
cell_type = []
for ds in asap_adata.uns['dataset_list']:
    
    f = hf.File(inpath+ds+'.h5ad','r')
    cell_ids = [x.decode('utf-8') for x in f['obs']['_index']]
    codes = list(f['obs']['cell_type']['codes'])
    cat = [x.decode('utf-8') for x in f['obs']['cell_type']['categories']]
    f.close()
    
    catd ={}
    for ind,itm in enumerate(cat):catd[ind]=itm
    
    cell_type = cell_type + [ [x,catd[y]] for x,y in zip(cell_ids,codes)]

df_ct = pd.DataFrame(cell_type,columns=['cell','celltype'])

df_cur = pd.read_csv(inpath+'tabula_sapiens_celltype_map.csv')
ctmap = {x:y for x,y in zip(df_cur['celltype'],df_cur['group'])}
df_ct['celltype2'] = [ctmap[x] if x in ctmap.keys() else 'others' for x in df_ct['celltype']]

selected = [x.split('@')[0] for x in asap_adata.obs.index.values]
df_ct = df_ct[df_ct['cell'].isin(selected)]
ctmap = {x:y for x,y in zip(df_ct['cell'],df_ct['celltype2'])}

 
asap_adata.obs['celltype'] = [ctmap[x] if x in ctmap.keys() else 'others' for x in selected]
asappy.plot_umap(asap_adata,col='celltype')

asap_adata.obs['celltype2'] = [ x.split('@')[1] for x in asap_adata.obs.index.values]
asappy.plot_umap(asap_adata,col='celltype2')


asap_adata.write('./results/'+sample+'.h5asapad')