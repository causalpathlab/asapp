import numpy as np
from sklearn.preprocessing import StandardScaler
import asapc

pb = '_pbulk.npz'
pbulkf = np.load(pb,allow_pickle=True)

K = 25

print('dc nmf model...nmf ')
nmf_model = asapc.ASAPdcNMF(pbulkf['pbulk'].T,K)
nmf = nmf_model.nmf()


scaler = StandardScaler()
scaled = scaler.fit_transform(nmf.beta_log)

preg_model = asapc.ASAPaltNMFPredict(pbulkf['pbulk'].T,scaled)
preg = preg_model.predict()

np.savez('_dcnmf_pbulk',
        beta = nmf.beta,
        beta_log = nmf.beta_log,
        theta = preg.theta,
        corr = preg.corr)