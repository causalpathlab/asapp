import numpy as np
import pandas as pd
import logging
from scipy.special import factorial
logger = logging.getLogger(__name__)



#### gradient descent #########

def loss(x, y, w, b):
	y_hat = np.exp(np.dot(x, w) + b)
	#error = np.square(y_hat - y).mean() / 2
	error = (y_hat - np.log(y_hat) * y).mean()
	return error
		

def grad(x, y, w, b):
	M, n = x.shape
	y_hat = np.exp( np.dot(x , w) + b)
	dw = ( np.dot(x.T , (y_hat - y))) / M
	db = (y_hat - y).mean()
	return dw, db

def gradient_descent(x, y, num_iter):
	w = np.zeros((x.shape[1],))
	alpha = 0.001
	b = 1
	
	for iter in range(num_iter):
		dw, db = grad(x, y, w, b)
		w -= alpha * dw 
		b -= alpha * db

	return w, b

#### coordinate descent least sqrs #########


def coordinate_descent(x,y,num_iter):
	x=(np.column_stack((np.ones(x.shape[0]),x)))
	x= x / np.sqrt(np.sum(np.square(x), axis=0))
	w = np.zeros(x.shape[1])
	
	for itr in range(num_iter):
		r =  x.dot(w) - y 
		for j in range(len(w)):
			w[j] = w[j] -  np.dot(x[:, j],r)
	return w

#### newton method #########

# For Lambda_i
def poisson_lambda(x,beta): return np.exp(-np.dot(x, beta))

# Log-likelihood function
def LL(x, y, beta): 
	return np.sum(-poisson_lambda(x, beta) + y * np.dot(x, beta) - np.log(factorial(y)))

# Gradient
def LL_D(x, y, beta):
	return np.dot((y-poisson_lambda(x, beta)), x)

# Hessian 
def LL_D2(x, y, beta):
	return -np.dot(np.transpose(poisson_lambda(x,beta).reshape(-1,1) * x), x)


def newton_opt(X,y,max_iter):
	X=(np.column_stack((np.ones(X.shape[0]),X)))
	beta = np.zeros((X.shape[1], ))
	for i in range(max_iter):
		beta_new = beta + np.dot(np.linalg.inv(LL_D2(X,y,beta)), LL_D(X,y,beta)) # Newton's update rule

		# stopping criteria
		if np.sum(np.absolute(beta-beta_new)) < 1e-12:
			# Update beta
			beta = beta_new
			break
		beta = beta_new
	return beta

