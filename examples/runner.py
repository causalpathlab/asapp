import sys
sys.path.insert(1, '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/asapp/')
import joblib

from util._io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from scannotation import ASAPP 
from data._dataloader import DataSet
from util import _topics


import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

from data import _sim 
from scipy import stats
from sklearn.metrics import mean_squared_error as mse


experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())



dl = DataSet()
dl.config = args
dl.initialize_path()
print(dl.inpath)
print(dl.outpath)


logging.basicConfig(filename=dl.outpath+'_model.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')




#########for sim data
# N = 1000
# K = 10
# P = 2000
# X = pd.read_csv(dl.inpath+'_X.csv.gz')
# dl.mtx = np.asmatrix(X)
# dl.rows = ['c_'+str(i) for i in range(N) ]
# dl.cols = ['g_'+str(i) for i in range(P) ]

####### for real data 
dl.initialize_data()
dl.load_data()

asap = ASAPP(adata=dl,tree_max_depth=10,factorization='VB', max_iter=50)
asap.factorize()
asap.predict(dl.mtx)

joblib.dump(asap.model,dl.outpath+'_model_vb.pkl')
logging.info('saved model.')

# asap = ASAPP(adata=dl,tree_min_leaf=1,tree_max_depth=10,factorization='MVB',max_iter=50,max_pred_iter=25,n_pass=50,batch_size=64)
# asap.factorize()
# asap.predict(dl.mtx)

# joblib.dump(asap.model,dl.outpath+'_model_mvb.pkl')
# logging.info('saved model.')
