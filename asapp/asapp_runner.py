from util._io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import fastsca
import logging
import _dcpnmf
np.random.seed(42)

experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sca = fastsca.FASTSCA()
sca.config = args

fn = sca.config.home + sca.config.experiment +sca.config.output + sca.config.sample_id+'/'+sca.config.sample_id


logging.basicConfig(filename=fn+'_model.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')



# for real data
# sca.initdata()
# sca.loaddata()
# print(sca.data.mtx.shape)

# for sim data
X = pd.read_csv(fn+'_X.csv.gz')
sca.data.mtx = np.asmatrix(X)


### batch
fastsca.run_dcasapp(sca,min_leaf=10,max_depth=10,n_components=10,max_iter=5,n_pass=50,mode='batch',save=fn)
fastsca.run_scNMF(sca,n_components=10,max_iter=5,n_pass=25,batch_size=256,mode='batch',save=fn+'_sc')

### all
# fastsca.run_dcasapp(sca,min_leaf=1,max_depth=10,n_components=10,max_iter=100,mode='all',save=fn)
# fastsca.run_scNMF(sca,n_components=10,max_iter=50,mode='all',save=fn+'_sc')