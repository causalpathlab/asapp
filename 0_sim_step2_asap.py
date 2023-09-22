######################################################
##### asap pipeline
######################################################


######################################################
##### single cell nmf
######################################################
import asappy
import sys


# sample = str(sys.argv[1])
sample = 'sim_r_1.0_s_10_sd_1'
print(sample)
asappy.create_asap_data(sample)

data_size = 25000
number_batches = 5
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

normalize_pb='lscale'
hvg_selection=False
gene_mean_z=10
gene_var_z=2
normalize_raw=None

asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_raw=normalize_raw,normalize_pb=normalize_pb,hvg_selection=hvg_selection,gene_mean_z=gene_mean_z,gene_var_z=gene_var_z)

asappy.asap_nmf(asap_object,num_factors=13)
asappy.save_model(asap_object)


#### nmf analysis
import anndata as an
from pyensembl import ensembl_grch38

asap_adata = an.read_h5ad('./results/'+sample+'.h5asap')

gn = []
for x in asap_adata.var.index.values:
    try:
        g = ensembl_grch38.gene_by_id(x.split('.')[0]).gene_name 
        gn.append(g)
    except:
        gn.append(x)

asap_adata.var.index = gn
asappy.leiden_cluster(asap_adata,k=10,mode='corr',resolution=0.1)
asappy.run_umap(asap_adata,distance='cosine',min_dist=0.1)


ct = [ x.replace('@'+sample,'') for x in asap_adata.obs.index.values]
ct = [ '-'.join(x.split('_')[2:]) for x in ct]
asap_adata.obs['celltype'] = ct
asap_adata.write_h5ad('./results/'+sample+'.h5asapad')