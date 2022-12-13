import numpy as np
import pandas as pd
from numpy.random import normal
np.random.seed(37)

###### probabilistic PCA using MLE and EM - Bishop chapter 12

def _log_likelihood(X,W,sigma2): 
	# Bishop - 12.44
	C = np.dot(W,W.T)+sigma2*np.eye(X.shape[1])
	S = 1/X.shape[0] * (X-X.mean(axis=0)).T @ (X-X.mean(axis=0))
	l = np.sum(-X.shape[0]/2 * (X.shape[1]*np.log(2*np.pi) + np.log(np.linalg.det(C) +np.trace(np.linalg.inv(C)@S)) ))
	return l 

def ppca(X,n_components,mode,n_iter=100,tol=1e-6):
	
	X = X - X.mean(axis=0)

	if mode == 'eigen':
		## eigen decomposition method
		S = np.dot(X.T,X) * 1/X.shape[0]
		eig_val,eig_vec = np.linalg.eig(S)
		eig_sort = np.argsort(eig_val)[::-1]

		#closed form solutions 
		# Bishop - 12.45/46
		sigma2 = 1/(X.shape[-1]- n_components) * np.sum(eig_val[eig_sort[-n_components:]])
		W = eig_vec[:,eig_sort[:n_components]] @ np.sqrt(np.diag(eig_val[eig_sort[:n_components]])-sigma2*np.eye(n_components))

		print('llik',_log_likelihood(X,W,sigma2))

	elif mode =='em':
		sigma2 = np.random.random()
		W = np.random.randn(X.shape[1], n_components)
		w_norm_diff = 1
		i = 0
		while i<n_iter and w_norm_diff>tol:
			W_old = W

			# e_step(X)
			# Bishop - 12.54/55
			M = W.T @ W +sigma2 * np.eye(n_components)
			M_inv = np.linalg.pinv(M)
			z = M_inv @ W.T @ X.T
			zzt = sigma2 * M_inv + z @ z.T

			# m_step(X)
			# Bishop - 12.56/57
			old_w = W
			W =(X.T @ z.T) @ np.linalg.pinv(zzt)
			sigma2 = np.trace(X +  X @ old_w @ M_inv @ W.T)/ (X.shape[0]*X.shape[1]) 

			w_norm_diff = np.linalg.norm(W-W_old)
			i += 1
			# print(i,_log_likelihood(X,W,sigma2)) ##em likelihood not working 
	
	# predictive distribution, Bishop - 12.41/42
	M_inv = np.linalg.pinv(W.T @ W + sigma2 * np.eye(n_components))
	X_transform = np.zeros((X.shape[0],n_components))
	X_transform_cov = sigma2 * M_inv
	X_transform_mean = np.dot(np.dot(M_inv,W.T),(X-X.mean(axis=0)).T)

	for i in range(X.shape[0]):
		X_transform[i] = np.random.multivariate_normal(X_transform_mean[:,i],X_transform_cov)
    
	return X_transform
