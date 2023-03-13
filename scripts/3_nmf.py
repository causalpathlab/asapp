import sys
import numpy as np
from asap.data.dataloader import DataSet
import asapc
import pandas as pd


inpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/data/simdata/simdata'
outpath = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/simdata/simdata'
pb = '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/result/simdata/simdata_pbulk.npz'
# inpath = sys.argv[1]
# outpath = sys.argv[2]
# pb = sys.argv[3]

dl = DataSet(inpath,outpath,data_mode='sparse',data_ondisk=False)
dl.initialize_data()
dl.load_data()

pbulkf = np.load(pb,allow_pickle=True)

K = 10

######## alt nmf model 

print('alt nmf model...nmf ')
nmf_model = asapc.ASAPaltNMF(pbulkf['pbulk'].T,K)
nmf = nmf_model.nmf()

print('alt nmf model...predict ')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(nmf.beta_log)

reg_model = asapc.ASAPaltNMFPredict(dl.mtx.T,scaled)
reg = reg_model.predict()

print('alt nmf model...saving ')

np.savez(outpath+'_altnmf',
        beta = nmf.beta,
        beta_log = nmf.beta_log,
        theta = reg.theta,
        corr = reg.corr)


######## dc nmf model 

import threading
from multiprocessing import Pool, Process, Queue

K=10

print('dc nmf model...nmf ')
nmf_model = asapc.ASAPdcNMF(pbulkf['pbulk'].T,K)
nmf = nmf_model.nmf()

print('dc nmf model...predict using alt ')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(nmf.beta_log)

reg_model = asapc.ASAPaltNMFPredict(dl.mtx.T,scaled)
reg = reg_model.predict()

print('alt nmf model...saving ')

np.savez(outpath+'_dcnmf',
        beta = nmf.beta,
        beta_log = nmf.beta_log,
        theta = reg.theta,
        corr = reg.corr)


######### parallel if using dc predict ###########

# print('dc nmf model...predict using dc ')

# results = []
# def collect_results(result):
#     results.extend(result)

# def _worker(rows,mtx):
#         model_predict = asapc.ASAPdcNMFPredict(mtx,nmf.beta_a,nmf.beta_b)
#         asap_res = model_predict.predict()
#         df = pd.DataFrame(asap_res.theta)
#         df['rows']=rows
#         return df.values.tolist()


# n_cores=10
# pool = Pool(n_cores)
# x = np.linspace(0,dl.mtx.T.shape[1],n_cores,dtype=int)
# blocks=[]
# for i in range(len(x)):
#         if i < len(x)-1:
#                 blocks.append((x[i],x[i+1]-1))


# for start,end in blocks:
#         pool.apply_async(_worker, args=(dl.rows[start:end],dl.mtx.T[:,start:end]), callback=collect_results)
# pool.close()
# pool.join()

# pd.DataFrame(results).to_csv(outpath+'_dctheta.csv.gz',index=False, compression='gzip')
