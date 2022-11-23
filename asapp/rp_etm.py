import pandas as pd
from topicmodel import scetm, _runner_etm
fpath = '/home/BCCRC.CA/ssubedi/projects/experiments/fastsca/result/tnbc/tnbc_rp'
print(fpath)
df = pd.read_csv(fpath+"_bulk.csv.gz")

rp_etm = scetm.SCETM()
rp_etm.data.mtx_indptr = fpath+'.indptr.npy'
rp_etm.data.mtx_indices = fpath+'.indices.npy'
rp_etm.data.mtx_data = fpath+'.data.npy'
rp_etm.data.rows = list([ 'r'+str(i) for i in range(df.shape[0])])  
rp_etm.data.cols =  list([ 'c'+str(i) for i in range(df.shape[1])])
rp_etm.etm.model_id = fpath
# _runner_etm.run_model(rp_etm)



#########

from util._io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import fastsca
import logging

experiment = '/projects/experiments/fastsca/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sca = fastsca.FASTSCA()
sca.config = args
sca.initdata()
sca.loaddata()
sca.data.mtx.shape
sca = fastsca.FASTSCA()
sca.config = args
sca.initdata()
sca.loaddata()
sca.data.mtx.shape

#########

eval_fpath = '/home/BCCRC.CA/ssubedi/projects/experiments/fastsca/data/tnbc/tnbc'
print(fpath)
ev_etm = scetm.SCETM()
ev_etm.data.mtx_indptr = eval_fpath+'.indptr.npy'
ev_etm.data.mtx_indices = eval_fpath+'.indices.npy'
ev_etm.data.mtx_data = eval_fpath+'.data.npy'
ev_etm.data.rows = sca.data.rows
ev_etm.data.cols =  sca.data.cols
ev_etm.etm.model_id = fpath
_runner_etm.eval_model(rp_etm,ev_etm)

_runner_etm.generate_results(ev_etm)