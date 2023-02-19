import numpy as np
from scipy import stats
import asapc as asap

N = 1000
K = 20
M = 1000

Theta = stats.gamma.rvs(0.5, scale=0.1, size=(N,K))
Beta = stats.gamma.rvs(0.5, scale=0.1, size=(M,K))
X = stats.poisson.rvs(Theta.dot(Beta.T))

model = asap.ASAP(X,K)
res = model.run()