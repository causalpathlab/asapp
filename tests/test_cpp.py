
import sys
sys.path.insert(1, '/home/BCCRC.CA/ssubedi/projects/experiments/asapp/asap/')
import numpy as np
from scipy import stats
import asapc as asap

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

N = 100
K = 5
M = 200
Theta = stats.gamma.rvs(0.5, scale=0.1, size=(N,K))
Beta = stats.gamma.rvs(0.5, scale=0.1, size=(M,K))
X = stats.poisson.rvs(Theta.dot(Beta.T))



model = asap.ASAPNMFAlt(X.T,K)
nmf = model.run()