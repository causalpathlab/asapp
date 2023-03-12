import sys
import numpy as np
from asap.data.dataloader import DataSet
import asapc


inpath = sys.argv[1]
outpath = sys.argv[2]

dl = DataSet(inpath,outpath,data_mode='sparse',data_ondisk=False)
dl.initialize_data()
dl.load_data()

pbulkf = np.load(sys.argv[3],allow_pickle=True)

K = 10

######## alt nmf model 

print('alt nmf model...nmf ')
nmf_model = asapc.ASAPaltNMF(pbulkf['pbulk'].T,K)
nmf = nmf_model.nmf()

print('alt nmf model...predict ')
reg_model = asapc.ASAPaltNMFPredict(dl.mtx.T,nmf.beta_log)
reg = reg_model.predict()

print('alt nmf model...saving ')

np.savez(outpath+'_altnmf',
        beta = nmf.beta,
        theta = reg.theta,
        corr = reg.corr)


######## dc nmf model 

print('dc nmf model...nmf ')
nmf_model = asapc.ASAPdcNMF(pbulkf['pbulk'].T,K)
nmf = nmf_model.nmf()

print('dc nmf model...predict ')
model_predict = asapc.ASAPdcNMFPredict(dl.mtx.T,nmf.beta_a,nmf.beta_b)
asap_res = model_predict.predict()

print('dc nmf model...saving ')
np.savez(outpath+'_dcnmf',
        beta = asap_res.beta,
        theta = asap_res.theta)


