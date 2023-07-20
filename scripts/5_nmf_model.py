from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.util import analysis

import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc


experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id

dl = DataSet(sample_in,sample_out,data_mode='sparse',data_ondisk=False)
dl.config = args
dl.initialize_data()
print(dl.inpath)
print(dl.outpath)


model = np.load(sample_out+'_dcnmf.npz')


top_genes = 10
df_beta = pd.DataFrame(np.exp(model['beta']).T)
df_beta.columns = dl.cols
df_top = analysis.get_topic_top_genes(df_beta.iloc[:,:],top_n=top_genes)
df_top.to_csv(dl.outpath+'_beta_top_genes.csv.gz',index=False,compression='gzip')




df_theta = pd.DataFrame(model['theta'])
df_theta.index = dl.rows
df_theta['topic'] = [ x.split('_')[1] for x in df_theta.index]
dfh= df_theta.groupby('topic').sample(n=50, random_state=1)
dfh.index = dfh.index.set_names(['cell'])
dfh.reset_index(inplace=True)
dfh.to_csv(dl.outpath+'_theta_sample_topic.csv.gz',index=False,compression='gzip')