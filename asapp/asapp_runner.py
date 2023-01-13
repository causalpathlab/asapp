from util._io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from scannotation import ASAPP
from util._dataloader import DataSet
from util import _topics


import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc


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




## for sim data
N=1000
K=10
P=2000
X = pd.read_csv(dl.inpath+'_X.csv.gz')
dl.mtx = np.asmatrix(X)
dl.rows = ['c_'+str(i) for i in range(N) ]
dl.cols = ['g_'+str(i) for i in range(P) ]


asap = ASAPP(adata=dl)
asap.generate_bulk()
asap.factorize()
asap.save_model()

dl.outpath += '_sc'
asapsc = ASAPP(adata=dl,experiment_mode='sc')
asapsc.factorize()
asapsc.save_model()