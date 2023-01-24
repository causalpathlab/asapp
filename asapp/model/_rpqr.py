import numpy as np
from scipy.linalg import qr
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

class RPQR():
    def __init__(self,mat,rp_mat,dc_mat):
        self.root = None
        self.mat = mat
        self.rp_mat = rp_mat
        self.dc_mat = dc_mat
    
    def get_psuedobulk(self):
                
        Z = np.dot(self.mat,self.rp_mat.T)
        Q, _ = qr(Z,mode='economic')
        Q[Q<0]=0
        return np.dot(Q.T,self.mat)

        # scaler = MinMaxScaler(feature_range=(0,1e6))
        # return scaler.fit_transform(pb)
    
def get_qr_basis(X,r):
    rp_mat = np.random.randn(X.shape[1],r)
    Z = np.dot(X,rp_mat)
    Q, _ = qr(Z,mode='economic')
    return Q

