import sys
sys.path.insert(1, '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/asapp/')

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
dl.initialize_data()
dl.load_data()
print(dl.inpath)
print(dl.outpath)

logging.basicConfig(filename=dl.outpath+'_model.log',
						format='%(asctime)s %(levelname)-8s %(message)s',
						level=logging.INFO,
						datefmt='%Y-%m-%d %H:%M:%S')

asap = ASAPP(adata=dl,tree_min_leaf=100,tree_max_depth=10, factorization='VB')
asap.factorize()
asap.save_model()

# dl = DataSet()
# dl.config = args
# dl.initialize_path()
# print(dl.inpath)
# print(dl.outpath)


# logging.basicConfig(filename=dl.outpath+'_model.log',
# 						format='%(asctime)s %(levelname)-8s %(message)s',
# 						level=logging.INFO,
# 						datefmt='%Y-%m-%d %H:%M:%S')




# ## for sim data
# N = 10100
# K = 10
# P = 15000

# H = stats.gamma.rvs(0.5, scale=0.1, size=(N,K))
# W = stats.gamma.rvs(0.5, scale=0.1, size=(P,K))
# X = stats.poisson.rvs(H.dot(W.T))

# dl.mtx = np.asmatrix(X)
# dl.rows = ['c_'+str(i) for i in range(N) ]
# dl.cols = ['g_'+str(i) for i in range(P) ]

# asap = ASAPP(adata=dl,tree_min_leaf=100,tree_max_depth=10, factorization='VB')
# asap.factorize()
# asap.save_model()
