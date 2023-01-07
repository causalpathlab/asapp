from util._io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import fastsca
import logging
import _pnmf,_dcpnmf,_dcpnmfb,_dcpnmfv2
import importlib
importlib.reload(_dcpnmf)
np.random.seed(42)

experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sca = fastsca.FASTSCA()
sca.config = args
sca.initdata()
sca.loaddata()
print(sca.data.mtx.shape)
fn = sca.config.home + sca.config.experiment +sca.config.output + sca.config.sample_id+'/'+sca.config.sample_id


logging.basicConfig(filename=fn+'_model.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

fastsca.run_dcasapp(sca,min_leaf=1000,max_depth=5,n_components=10,save=fn)
fastsca.run_scNMF(sca,fn+'_sc')


df_bd = pd.read_csv(fn+'_depth.csv.gz')
df_bf = pd.read_csv(fn+'_freq.csv.gz')
df_bbeta = pd.read_csv(fn+'_beta.csv.gz')
df_btheta = pd.read_csv(fn+'_theta.csv.gz')

df_scd = pd.read_csv(fn+'_sc_depth.csv.gz')
df_scf = pd.read_csv(fn+'_sc_freq.csv.gz')
df_scbeta = pd.read_csv(fn+'_sc_beta.csv.gz')
df_sctheta = pd.read_csv(fn+'_sc_theta.csv.gz')
df_btheta = df_btheta.rename(columns={'Unnamed: 0':'cell'})
df_bbeta = df_bbeta.rename(columns={'Unnamed: 0':'topic'})
df_sctheta = df_sctheta.rename(columns={'Unnamed: 0':'cell'})
df_scbeta = df_scbeta.rename(columns={'Unnamed: 0':'topic'})
