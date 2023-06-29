# import sys
# import numpy as np
# from asap.annotation import ASAPNMF
# from asap.data.dataloader import DataSet

# inpath = sys.argv[1]
# outpath = sys.argv[2]

# dl = DataSet(inpath,outpath)
# dl.initialize_data()
# dl.load_data()

# asap = ASAPNMF(adata=dl,tree_max_depth=10)
# asap.get_pbulk()
# np.savez(outpath, pbulk= asap.pbulk_mat)



from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.util import topics
from asap.annotation import ASAPNMF

import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

# import logging


experiment = '/projects/experiments/asapp/'
server = Path.home().as_posix()
experiment_home = server+experiment
experiment_config = read_config(experiment_home+'config.yaml')
args = namedtuple('Struct',experiment_config.keys())(*experiment_config.values())

sample_in = args.home + args.experiment + args.input+ args.sample_id +'/'+args.sample_id
sample_out = args.home + args.experiment + args.output+ args.sample_id +'/'+args.sample_id


dl = DataSet(sample_in,sample_out)

dl.initialize_data()
dl.add_batch_label([i.split('_')[1] for i in dl.barcodes])
dl.load_data()

asap = ASAPNMF(adata=dl,tree_max_depth=10)
asap.get_pbulk()


