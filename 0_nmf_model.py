
from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.factorize import ASAPNMF
from asap.util import analysis
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

from sklearn.preprocessing import StandardScaler
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

tree_max_depth = 10
num_factors = 20
batch_size = 10000
downsample_pseudobulk = True
downsample_size = 100
bn = 3

dl = DataSet(sample_in,sample_out)
sample_list = dl.get_dataset_names()
dl.initialize_data(sample_list,batch_size)

asap = ASAPNMF(adata=dl,tree_max_depth=tree_max_depth,num_factors=num_factors,downsample_pbulk=downsample_pseudobulk,downsample_size=downsample_size,num_batch=bn)
asap.generate_pseudobulk()
asap.filter_pbulk(5) 

asap.get_psuedobulk_batchratio('@')
analysis.plot_pbulk_batchratio(asap.pbulk_batchratio,dl.outpath+'_pb_batch_ration.png')

asap.run_nmf()