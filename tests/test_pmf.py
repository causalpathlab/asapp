from model import _dcpmf
import numpy as np
np.random.seed(42)

from scipy import stats
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pylab as plt

import pytest
from numpy.testing import assert_allclose

np.random.seed(42)

def generate_data(N,K,M,mode):    

    if mode=='block':    
        from util import _sim as _sim
        H = _sim.generate_H(N, K)
        W = _sim.generate_W(M, K)
        X = stats.poisson.rvs(H.dot(W.T))

    else:
        H = stats.gamma.rvs(0.5, scale=0.1, size=(N,K))
        W = stats.gamma.rvs(0.5, scale=0.1, size=(M,K))
        X = stats.poisson.rvs(H.dot(W.T))

    return H,W,X

def dcpmf(X,K,max_iter):
    pmf = _dcpmf.DCPoissonMF(n_components=K,max_iter=max_iter,verbose=True)
    pmf.fit(X)
    return pmf

def dcpmf_batch(X,K,max_iter,n_pass):
    pmf = _dcpmf.DCPoissonMFBatch(n_components=K,max_iter=max_iter,n_pass=n_pass,verbose=True)
    pmf.fit(X)
    pmf.transform(X)
    return pmf

def dcpmf_membatch(X,K,max_iter,n_pass):
    pmf = _dcpmf.DCPoissonMFMemBatch(n_components=K,max_iter=max_iter,n_pass=n_pass,verbose=True)
    pmf.fit(X)
    pmf.transform(X)
    return pmf

def test_dcpmf():

    N=100
    K=5
    M=200
    max_iter = 200
    n_pass = 200

    H,W,X = generate_data(N,K,M,mode='bloc')

    pmf = dcpmf(X,K,max_iter)
    pmfb = dcpmf_batch(X,K,max_iter,n_pass)
    pmfmb = dcpmf_membatch(X,K,max_iter,n_pass)

    # assert_allclose(H,pmf.Etheta,rtol=1e-4)

    print('all-->')
    print("MSE W:", mse(W, pmf.Ebeta.T))
    print("MSE H:", mse(H, pmf.Etheta))

    print('batch-online-->')
    print("MSE W:", mse(W, pmfb.Ebeta.T))
    print("MSE H:", mse(H, pmfb.Etheta))

    print('batch-mem-->')
    print("MSE W:", mse(W, pmfmb.Ebeta.T))
    print("MSE H:", mse(H, pmfmb.Etheta))

    plt.plot(pmf.bound,'r',label='all')
    plt.plot(pmfb.bound,'b',label='batch-online')
    plt.plot(pmfmb.bound,'g',label='batch-mem')
    plt.legend()
    plt.savefig('test_pmf.png')


test_dcpmf()

