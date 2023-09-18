sample = 'pbmc'
import asappy
# asappy.create_asap_data(sample)
data_size = 25000
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

normalization_raw='unitnorm'
normalization_pb='scaler'

asappy.generate_pseudobulk(asap_object,tree_depth=10,normalization_raw=normalization_raw,normalization_pb=normalization_pb)

mtx = asap_object.adata.uns['pseudobulk']['pb_data'].T


from asappy.preprocessing.normalize import normalize_pb

outpath = asap_object.adata.uns['inpath']+'pb_'
asappy.plot_dmv_distribution(mtx,outpath)
asappy.plot_blockwise_totalgene_bp(mtx,outpath,'mean')
asappy.plot_blockwise_totalgene_depth_sp(mtx,outpath,mode='mean')

mtx_n = normalize_pb(mtx.T,method='scaler')

asappy.plot_dmv_distribution(mtx_n.T,outpath+'_n')
asappy.plot_blockwise_totalgene_bp(mtx_n.T,outpath+'_n','mean')
asappy.plot_blockwise_totalgene_depth_sp(mtx_n.T,outpath+'_n',mode='mean')
