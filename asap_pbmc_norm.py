sample = 'pbmc'
import asappy
# asappy.create_asap_data(sample)
data_size = 25000
number_batches = 1
asap_object = asappy.create_asap_object(sample=sample,data_size=data_size,number_batches=number_batches)

normalization_raw='unitnorm'
normalization_pb='rtmqsr'

asappy.generate_pseudobulk(asap_object,tree_depth=10,normalization_raw=normalization_raw,normalization_pb=normalization_pb)

from asappy.preprocessing.normalize import normalize_pb

outpath = asap_object.adata.uns['inpath']+'pb_'

mtx = asap_object.adata.uns['pseudobulk']['pb_data'].T
asappy.plot_dmv_distribution(mtx,outpath)
asappy.plot_blockwise_totalgene_bp(mtx,outpath,'mean')
asappy.plot_blockwise_totalgene_depth_sp(mtx,outpath,mode='mean')

def pltmv(mtx,outpath):
    import matplotlib.pylab as plt
    import seaborn as sns
    import numpy as np
    m = mtx.mean(0)
    v = mtx.mean(0)
    plt.plot(np.log10(m),v,'o')
    plt.savefig(outpath+'mv.png');plt.close()
pltmv(mtx,outpath)

pd.cut(mtx.mean(0),bins=10).value_counts()


mtxn = normalize_pb(mtx,'rtmsqr')

asappy.plot_dmv_distribution(mtxn,outpath+'_n')
asappy.plot_blockwise_totalgene_bp(mtxn,outpath+'_n','mean')
asappy.plot_blockwise_totalgene_depth_sp(mtxn,outpath+'_n',mode='mean')
pltmv(mtxn,outpath)

