import numpy as np
from scipy.special import factorial


#### logistic regression newton method #########
def lr_sigmoid(z): return 1 / (1 + np.exp(-z))

def lr_logl( x,beta,y, p):
	# return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean() ## OR further simplified
	return -(y * np.dot(x,beta) - np.log(1 + np.exp(np.dot(x,beta)))).mean()

def lr_logl_d(x, y, p):
	return np.dot(x.T, (y-p))

def lr_logl_d2(x, p):
    W = np.diag(p * (1 - p))
    return -(x.T.dot(W).dot(x))

def lr_optimizer(X,y,max_iter):
	llik = []
	intercept = np.ones((X.shape[0], 1))
	X = np.concatenate((intercept, X), axis=1)
	beta = np.zeros(X.shape[1])
	for i in range(max_iter):

		z = np.dot(X,beta)
		p = lr_sigmoid(z)
		gradient = lr_logl_d(X,y,p)
		hessian = lr_logl_d2(X,p) 

		beta = beta - np.dot(np.linalg.inv(hessian), gradient) 
		llik.append(lr_logl(X,beta,y,p))
		if np.isnan(llik[len(llik)-1]):
			break
		elif llik[len(llik)-1] > llik[len(llik)-2]:
			break

	return beta,llik

#### poisson regression newton method #########

def p_lambda(x,beta): return np.exp(np.dot(x, beta))

def p_logl(x, y, beta): 
	return np.sum(  y * np.dot(x, beta) - p_lambda(x, beta) - np.log(factorial(y)))

def p_logl_d(x, y, beta):
	return np.dot((y-p_lambda(x, beta)),x)

def p_logl_d2(x, y, beta):
	return -np.dot(np.transpose(p_lambda(x,beta).reshape(-1,1) * x), x)

def poissonr_optimizer(X,y,max_iter):
	llik = []
	X=(np.column_stack((np.ones(X.shape[0]),X)))
	beta = np.zeros((X.shape[1], ))
	for i in range(max_iter):
		beta_new = beta - np.dot(np.linalg.inv(p_logl_d2(X,y,beta)), p_logl_d(X,y,beta)) 
		if np.sum(np.absolute(beta-beta_new)) < 1e-12:
			beta = beta_new
			break
		beta = beta_new
		llik.append(p_logl(X,y,beta))

	return beta,llik




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