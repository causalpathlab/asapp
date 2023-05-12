


import sys
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/')

import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet

from asap.annotation import ASAPNMF

import asapc

from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.util import topics

import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

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

dl = DataSet('pbmc',sample_in,sample_out)

dl.initialize_data()
dl.add_batch_label([i.split('_')[1] for i in dl.barcodes])
dl.load_data()



import rpy2.robjects as ro
import rpy2.robjects.packages as rp
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


### naive bayes prob estimation
nb = ro.packages.importr('naivebayes')
laplace = 0.5

nr,nc = dl.mtx.T.shape
ro.r.assign("M", ro.r.matrix(dl.mtx.T, nrow=nr, ncol=nc))
ro.r('colnames(M) <- paste0("V", seq_len(ncol(M)))')
ro.r.assign('laplace',0.5)
ro.r.assign('N',np.array(dl.batch_label))
ro.r('pnb <- poisson_naive_bayes(x=M,y=N,laplace=laplace)')

probs = dict(ro.r('coef(pnb)').items())
df_probs = pd.DataFrame(probs)
sns.scatterplot(x='3k',y='4k',data=df_probs)
plt.savefig('prob.png');plt.close()

df_probs.index = dl.genes

sns.scatterplot(x='3k',y='4k',data=df_probs.loc[df_probs['4k']>10.0])
plt.savefig('prob2.png');plt.close()

### prob weighting


X_rows = dl.mtx.shape[0]
tree_max_depth = 10

rp_mat = []
for _ in range(tree_max_depth):
    rp_mat.append(np.random.normal(size = (X_rows,1)).flatten())                      
rp_mat = np.asarray(rp_mat)



asap = ASAPNMF(adata=dl,data_chunk=20000)
asap.get_pbulk()



###import sys
sys.path.append('/home/BCCRC.CA/ssubedi/projects/experiments/asapp/')

import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet

from asap.annotation import ASAPNMF

import asapc

from asap.util.io import read_config
from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
from asap.data.dataloader import DataSet
from asap.util import topics

import matplotlib.pylab as plt
import seaborn as sns
import colorcet as cc

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

dl = DataSet('pbmc',sample_in,sample_out)

dl.initialize_data()
dl.add_batch_label([i.split('_')[1] for i in dl.barcodes])
dl.load_data()
import numpy as np
import pandas as pd
from scipy.linalg import qr
import logging
logger = logging.getLogger(__name__)

import annoy
import random

class ApproxNN():
    def __init__(self, data, labels):
        self.dimension = data.shape[1]
        self.data = data.astype('float32')
        self.labels = labels

    def build(self, number_of_trees=50):
        self.index = annoy.AnnoyIndex(self.dimension,'angular')
        for i, vec in enumerate(self.data):
            self.index.add_item(i, vec.tolist())
        self.index.build(number_of_trees)

    def query(self, vector, k):
        indexes = self.index.get_nns_by_vector(vector.tolist(),k)
        return [self.labels[i] for i in indexes]

def get_models(mtx,bindexd):
    model_list = {}
    for batch in bindexd.keys():        
        model_ann = ApproxNN(mtx[:,bindexd[batch]].T,bindexd[batch])
        model_ann.build()
        model_list[batch] = model_ann
    return model_list



def get_rp(mtx,rp_mat,batch_label):

    Z = np.dot(rp_mat,mtx).T

    ## no batch correction 
    # logger.info('Randomized QR factorized pseudo-bulk')    
    Q, _ = qr(Z,mode='economic')
    Q = (np.sign(Q) + 1)/2


    ## batch correction 
    # logger.info('Randomized QR factorized pseudo-bulk with batch correction')    
    # b_mat = []
    # for b in list(set(batch_label)):
    #       b_mat.append([ 1 if x == b else 0 for x in batch_label])
    # b_mat = np.array(b_mat).T
    
    # u_batch, _, _ = np.linalg.svd(b_mat,full_matrices=False)
    # Zres = Z - u_batch@u_batch.T@Z
    # Q, _ ,_ = np.linalg.svd(Zres, full_matrices=False)
    # Q = (np.sign(Q) + 1)/2
    
    df = pd.DataFrame(Q,dtype=int)
    df['code'] = df.astype(str).agg(''.join, axis=1)
    df = df.reset_index()
    df = df[['index','code']]
    return df.groupby('code').agg(lambda x: list(x)).reset_index().set_index('code').to_dict()['index']


#### 

def pnb_estimation(mtx,batch_label):

    import rpy2.robjects as ro
    import rpy2.robjects.packages as rp
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    ro.packages.importr('naivebayes')

    nr,nc = mtx.T.shape
    ro.r.assign("M", ro.r.matrix(mtx.T, nrow=nr, ncol=nc))
    ro.r('colnames(M) <- paste0("V", seq_len(ncol(M)))')
    ro.r.assign('laplace',0.5)
    ro.r.assign('N',np.array(batch_label))
    ro.r('pnb <- poisson_naive_bayes(x=M,y=N,laplace=laplace)')

    return pd.DataFrame(dict(ro.r('coef(pnb)').items()))
mtx = dl.mtx
batch_label = dl.batch_label
X_rows = dl.mtx.shape[0]
tree_max_depth = 10

rp_mat = []
for _ in range(tree_max_depth):
    rp_mat.append(np.random.normal(size = (X_rows,1)).flatten())                      
rp_mat = np.asarray(rp_mat)
df_pnb = pnb_estimation(mtx,batch_label)
pbulkd = get_rp(mtx,rp_mat,batch_label)
batches = list(set(batch_label))


pbulk = {}

for key,vlist in pbulkd.items():
    
    if len(vlist) <= len(batches):
        continue

    else:
        updated_pb = []   
        for value in vlist:
            pb = mtx[:,value]
            col = []
            for b in batches:
                    col.append( -pb*np.log(df_pnb[b].values)+df_pnb[b].values )
            ci = batches.index(batch_label[value])
            col = np.array(col)
            w = col[ci]/col.sum(0)  
            updated_pb.append((pb/w)/(1/w))

    pbulk[key] = np.array(updated_pb).sum(0)


df_pnb.sum()
plt.scatter(df_pnb['3k'],df_pnb['4k'])
#plt.savefig('pnb.png');plt.close()

cd experiments/asapp/examples/
plt.savefig('pnb.png');plt.close()
plt.scatter(df_pnb['3k'],df_pnb['4k'],s=1)
plt.savefig('pnb.png');plt.close()
plt.scatter(df_pnb['3k'],df_pnb['4k'],s=5)
plt.savefig('pnb.png');plt.close()
col
plt.scatter(col[0],col[1],s=5)
plt.savefig('pnb_prob.png');plt.close()



### inverse probability score 