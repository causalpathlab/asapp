from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.util import topics

import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

from asap.data.dataloader import DataSet


experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id

dl = DataSet('pbmc',sample_in,sample_out)
dl.initialize_data()

print(dl.inpath)
print(dl.outpath)

dl.load_data()

from asap.annotation import ASAPNMF
asap = ASAPNMF(adata=dl)
asap.get_pbulk()