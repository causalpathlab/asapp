from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.util import topics
from asap.annotation import ASAPNMF
# import asapc
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc
from sklearn.preprocessing import StandardScaler
# import logging


experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id


dl = DataSet(sample_in,sample_out)
sample_list = dl.get_samplenames()
dl.initialize_data(sample_list,1000000)


# df=pd.read_csv('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/data/simdata/simdata_bl.csv')
# dl.add_batch_label([i.split('@')[1] for i in dl.barcodes])
dl.add_batch_label([i.split('_')[1] for i in dl.barcodes])
# dl.add_batch_label([i.split('-')[1] for i in dl.barcodes])
# dl.add_batch_label(df.x.values)

asap = ASAPNMF(adata=dl,tree_max_depth=10)
asap.get_pbulk()


pb_model = asapc.ASAPpb(asap.ysum,asap.zsum,asap.delta, asap.n_bs,asap.n_bs/asap.n_bs.sum(0),asap.size) 
pb_res = pb_model.generate_pb()
inpath = sample_in
outpath = sample_out
np.savez(outpath+'_pbulk', pbulk= pb_res.pb,pb_batch=pb_res.pb_batch,batch_effect=pb_res.batch_effect)

pbulkf = np.load(outpath+'_pbulk.npz')
pbulk = np.log1p(pbulkf['pbulk'])
K = 10

# ######## alt nmf model 

# logging.info('alt nmf model...nmf ')

# nmf_model = asapc.ASAPaltNMF(pbulk,K)
# nmf = nmf_model.nmf()

# logging.info('alt nmf model...predict ')


# scaler = StandardScaler()
# scaled = scaler.fit_transform(nmf.beta_log)

# reg_model = asapc.ASAPaltNMFPredict(dl.mtx,scaled)
# reg = reg_model.predict()

# logging.info('alt nmf model...saving ')

# np.savez(outpath+'_altnmf',
#         beta = nmf.beta,
#         beta_log = nmf.beta_log,
#         theta = reg.theta,
#         corr = reg.corr)



# preg_model = asapc.ASAPaltNMFPredict(pbulk,scaled)
# preg = preg_model.predict()

# np.savez(outpath+'_altnmf_pbulk',
#         beta = nmf.beta,
#         beta_log = nmf.beta_log,
#         theta = preg.theta,
#         corr = preg.corr)



######## dc nmf model 

# logging.info('dc nmf model...nmf ')
nmf_model = asapc.ASAPdcNMF(pbulk,K)
nmf = nmf_model.nmf()

# logging.info('dc nmf model...predict using alt ')


##TAKE REGRESSION 
# x is delta_db/pb_res.batch_effect and y is nmf.beta_log

u_batch, _, _ = np.linalg.svd(pb_res.batch_effect,full_matrices=False)
nmf_beta_log = nmf.beta_log - u_batch@u_batch.T@nmf.beta_log


scaler = StandardScaler()
scaled = scaler.fit_transform(nmf_beta_log)

reg_model = asapc.ASAPaltNMFPredict(dl.mtx,scaled)
reg = reg_model.predict()

# logging.info('dc nmf model...saving ')

np.savez(outpath+'_dcnmf',
        beta = nmf.beta,
        beta_log = nmf.beta_log,
        theta = reg.theta,
        corr = reg.corr)


#### predict pbulk 
# preg_model = asapc.ASAPaltNMFPredict(pbulk,scaled)
# preg = preg_model.predict()

# logging.info('dc nmf model...saving ')

# np.savez(outpath+'_dcnmf_pbulk',
#         beta = nmf.beta,
#         beta_log = nmf.beta_log,
#         theta = preg.theta,
#         corr = preg.corr)

