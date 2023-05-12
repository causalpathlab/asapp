
import sys
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/')

import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet

from asap.annotation import ASAPNMF

import asapc

from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.util import topics

import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

import logging


experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id

logging.basicConfig(filename=sample_out+'_model.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

dl = DataSet('pbmc',sample_in,sample_out)

dl.initialize_data()
dl.add_batch_label([i.split('_')[1] for i in dl.barcodes])
dl.load_data()

asap = ASAPNMF(adata=dl,data_chunk=20000)
asap.get_pbulk()

inpath = sample_in
outpath = sample_out

np.savez(outpath+'_pbulk', pbulk= asap.pbulk_mat)


pbulkf = np.load(outpath+'_pbulk.npz')
pbulk = np.log1p(pbulkf['pbulk'])
K = 10

######## alt nmf model 

logging.info('alt nmf model...nmf ')

nmf_model = asapc.ASAPaltNMF(pbulk,K)
nmf = nmf_model.nmf()

logging.info('alt nmf model...predict ')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(nmf.beta_log)

reg_model = asapc.ASAPaltNMFPredict(dl.mtx,scaled)
reg = reg_model.predict()

logging.info('alt nmf model...saving ')

np.savez(outpath+'_altnmf',
        beta = nmf.beta,
        beta_log = nmf.beta_log,
        theta = reg.theta,
        corr = reg.corr)



# preg_model = asapc.ASAPaltNMFPredict(pbulk,scaled)
# preg = preg_model.predict()

# np.savez(outpath+'_altnmf_pbulk',
#         beta = nmf.beta,
#         beta_log = nmf.beta_log,
#         theta = preg.theta,
#         corr = preg.corr)



######## dc nmf model 

logging.info('dc nmf model...nmf ')
nmf_model = asapc.ASAPdcNMF(pbulk,K)
nmf = nmf_model.nmf()

logging.info('dc nmf model...predict using alt ')

scaler = StandardScaler()
scaled = scaler.fit_transform(nmf.beta_log)

reg_model = asapc.ASAPaltNMFPredict(dl.mtx,scaled)
reg = reg_model.predict()

logging.info('dc nmf model...saving ')

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

