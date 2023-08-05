|

# scaler = StandardScaler()
# df_corr = pd.DataFrame(scaler.fit_transform(model['predict_corr']))
df_corr = pd.DataFrame(model['predict_corr'])
df_corr.index = model['predict_barcodes']

batch_label = ([x.split('@')[1] for x in df_corr.index.values])
# batches = set(batch_label)


# for i,b in enumerate(batches):
#     indxs = [i for i,v in enumerate(batch_label) if v ==b]
#     dfm = df_corr.iloc[indxs,:]
#     scaler = MinMaxScaler()
#     dfm = scaler.fit_transform(dfm)

#     if i ==0:
#         upd_indxs = indxs
#         upd_df = dfm
#     else:
#         upd_indxs += indxs
#         upd_df = np.vstack((upd_df,dfm))

# batch_label = np.array(batch_label)[upd_indxs]
# df_corr = pd.DataFrame(upd_df)
# df_corr.index = np.array(model['predict_barcodes'])[upd_indxs]



######### minibatch for visualization

# import random 

# N = 1000
# minib_i = random.sample(range(0,df_corr.shape[0]),N)
# df_corr = df_corr.iloc[minib_i,:]


## assign ids
df_umap= pd.DataFrame()
df_umap['cell'] =[x.split('@')[0] for x in df_corr.index.values]

# assign batch
batch_label = ([x.split('@')[1] for x in df_corr.index.values])
df_umap['batch'] = batch_label

# assign topic
kmeans = KMeans(n_clusters=num_factors, init='k-means++',random_state=0).fit(df_corr)
df_umap['asap_topic'] = kmeans.labels_

## assign celltype
# inpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/data/tabula_sapiens/'
# cell_type = []
# for ds in dl.dataset_list:
    
#     f = hf.File(inpath+ds+'.h5ad','r')
#     cell_ids = [x.decode('utf-8') for x in f['obs']['_index']]
#     codes = list(f['obs']['cell_type']['codes'])
#     cat = [x.decode('utf-8') for x in f['obs']['cell_type']['categories']]
#     f.close()
    
#     catd ={}
#     for ind,itm in enumerate(cat):catd[ind]=itm
    
#     cell_type = cell_type + [ [x,catd[y]] for x,y in zip(cell_ids,codes)]

# df_ct = pd.DataFrame(cell_type,columns=['cell','celltype'])


# df_umap = pd.merge(df_umap,df_ct,on='cell',how='left')
# df_umap = df_umap.drop_duplicates(subset='cell',keep=False)

# select_ct = list(df_umap.celltype.value_counts()[:10].index)
# select_ct = ['T', 'B', 'NK', 'monocyte' 'macrophage', 'plasma' , 'neutrophil' ]

# ct_map = {}
# for ct in select_ct:
#     for v in df_umap.celltype.values:
#         if ct in v:
#             ct_map[v]=ct

# df_umap['celltype'] = [ ct_map[x]+'_'+y if  x  in ct_map.keys() else 'others' for x,y in zip(df_umap['celltype'],df_umap['batch'])]


#### fix number of cells in data

# keep_index = np.where(np.isin(np.array([x.split('@')[0] for x in df_corr.index.values]), df_umap['cell'].values))[0]
# df_corr = df_corr.iloc[keep_index,:]

########### pre bc

df_umap[['umap_1','umap_2']] = analysis.get2dprojection(df_corr.to_numpy())
analysis.plot_umaps(df_umap,dl.outpath+'_pre_batchcorrection.png')

# df_umap.to_csv(dl.outpath+'_prebc_umap.csv.gz',compression='gzip')

############## post bc

# from asap.util import batch_correction as bc 

# df_corr_sc = pd.DataFrame(bc.batch_correction_scanorama(df_corr.to_numpy(),np.array(batch_label),alpha=0.001,sigma=15))
# df_corr_sc.index = df_corr.index

# np.corrcoef(df_corr.iloc[:,0],df_corr_sc.iloc[:,0])


# df_umap_sc = df_umap[['cell','asap_topic','batch']]
# df_umap_sc[['umap_1','umap_2']] = analysis.get2dprojection(df_corr_sc.to_numpy())

# df_umap_sc.to_csv(dl.outpath+'_postbc_umap.csv.gz',compression='gzip')

### plots
# df_umap = pd.read_csv(dl.outpath+'_prebc_umap.csv.gz')
# df_umap_sc = pd.read_csv(dl.outpath+'_postbc_umap.csv.gz')

# df_umap = df_umap.drop(columns=['Unnamed: 0'])
# df_umap_sc = df_umap_sc.drop(columns=['Unnamed: 0'])

# analysis.plot_umaps(df_umap_sc,dl.outpath+'_post_batchcorrection.png')

