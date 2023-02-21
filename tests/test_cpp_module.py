# import numpy as np
# from scipy import stats
# import asapc as asap

# N = 1000
# K = 20
# M = 1000

# Theta = stats.gamma.rvs(0.5, scale=0.1, size=(N,K))
# Beta = stats.gamma.rvs(0.5, scale=0.1, size=(M,K))
# X = stats.poisson.rvs(Theta.dot(Beta.T))

# model = asap.ASAP(X,K)
# res = model.run()



import sys
sys.path.insert(1, '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/asap/python')
sys.path.insert(1, '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/asap')

from util._io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import logging
# from scannotation import ASAPP
import asapc as ASAPP
from data._dataloader import DataSet
from util import _topics


import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

from data import _sim 
from scipy import stats
from sklearn.metrics import mean_squared_error as mse

import joblib

experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

dl = DataSet(data_ondisk=False)
dl.config = args
dl.initialize_path()
dl.initialize_data()
print(dl.inpath)
print(dl.outpath)
dl.load_data()
K = 10
model = ASAPP.ASAP(dl.mtx.T,K)
dcpmf = model.run()