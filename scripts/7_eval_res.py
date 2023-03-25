from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import glob
import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc


experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'

# sample_out = sys.argv[1]

flist = []
for name in glob.glob(sample_out+'/*/_eval*.csv'):
    flist.append(name)

df = pd.DataFrame()
for f in flist:
    dfc = pd.read_csv(f)
    df = pd.concat([df,dfc],axis=0,ignore_index=True)


grps = ['mode','alpha','rho','depth','size']
df = df.groupby(grps).agg(['mean','std' ]).reset_index()
df.columns = df.columns.map('_'.join).str.strip('_')
df = df.drop(columns=['seed_mean','seed_std'])
df.to_csv(sample_out+'eval_result.csv',index=False)