######################################################
##### asap pipeline
######################################################


######################################################
##### single cell nmf
######################################################
import asappy
import sys


sample = str(sys.argv[1])
n_topics = int(sys.argv[2])
print(sample)

asappy.create_asap_data(sample)

data_size = 25000
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)
asappy.generate_pseudobulk(asap_object,tree_depth=10,normalize_pb='lscale',downsample_pseudobulk=False,pseudobulk_filter=False)
asappy.asap_nmf(asap_object,num_factors=n_topics)
asappy.save_model(asap_object)
