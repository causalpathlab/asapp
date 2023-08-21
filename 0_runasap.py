
import asappy

asap_object = asappy.create_asap('pbmc',data_size= 10000)

asappy.generate_pseudobulk(asap_object,tree_depth=10)

asappy.asap_nmf(asap_object,num_factors=10)