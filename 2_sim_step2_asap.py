######################################################
##### asap pipeline
######################################################


######################################################
##### single cell nmf
######################################################
import asappy
import sys


sample = str(sys.argv[1])
print(sample)
asappy.create_asap_data(sample)

n_topics = 7

data_size = 25000
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale',downsample_pseudobulk=False,pseudobulk_filter=False)

asappy.asap_nmf(asap_object,num_factors=n_topics)
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

asappy.leiden_cluster(asap_adata)
ct = [ x.replace('@'+sample,'') for x in asap_adata.obs.index.values]
ct = [ '-'.join(x.split('_')[1:]) for x in ct]
asap_adata.obs['celltype'] = ct
asap_adata.write_h5ad('./results/'+sample+'.h5asapad')