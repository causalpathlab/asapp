import numpy as np

## different least squares method 

## multiplicative update
def mu(A,k,n_iter,delta=1e-5,tol=0.1):
	W = np.random.rand(A.shape[0],k)
	H = np.random.rand(k,A.shape[1])
	loss = []
	for n in range(n_iter):

		W_TA = np.dot(W.T, A)
		W_TWH = np.dot(W.T,np.dot(W,H)+delta)
		for i in range(H.shape[0]):
			for j in range(H.shape[1]):
				H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]

		AH_T = np.dot(A,H.T)
		WHH_T = np.dot(W,np.dot(H,H.T)) + delta

		for i in range(W.shape[0]):
			for j in range(W.shape[1]):
				W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]

		frob_norm = np.linalg.norm(A - np.dot(W,H), 'fro')
		# print("iteration " + str(n + 1) + ": " + str(frob_norm))
		loss.append(frob_norm)
		if frob_norm<tol: break

	return {'W':W,'H':H,'loss':loss}


## alternating least squares
def als(A,k,n_iter,delta=1e-5,tol=0.1):
	
	from numpy.linalg import lstsq

	W = np.random.rand(A.shape[0],k)
	H = np.random.rand(k,A.shape[1])
	loss = []
	for n in range(n_iter):
		
		'''
		lstsq(a,b)
		a @x = b, minimizes the euclidean 2-norm ||b-ax ||
		(W @x = A)
		'''
		
		'''idea is basis matrix is m x k and k x 1 is beta vector where 1 is each n/cell 
		H = np.dot((np.dot(np.linalg.inv(np.dot(W.T,W)),W.T)),A)
		'''
		
		H = lstsq(W,A,rcond=-1)[0]
		H[H<0] = 0

		'''idea is basis matrix is n x k and  k x 1 is beta vector where 1 is each m/gene
		W = np.dot((np.dot(np.linalg.inv(np.dot(H,H.T)),H)),A.T).T
		'''
		W = lstsq(H.T,A.T,rcond=-1)[0].T
		W[W<0] = 0

		frob_norm = np.linalg.norm(A - np.dot(W,H), 'fro')
		# print("iteration " + str(n + 1) + ": " + str(frob_norm))
		loss.append(frob_norm)
		if frob_norm<tol: break

	return {'W':W,'H':H,'loss':loss}


## alternating least squares with active set method

def active_set_solver(B,y,eps=1e-5):

	from numpy.linalg import lstsq

	'''
	B is m x n, y is m x 1, and g is n x 1, 
	solve a least squares problem ||Bg - y||_2.	
	
	'''
	n = B.shape[1]
	E = np.ones(n) ## keep track of active n, initialize as all active
	S = np.zeros(n) ## keep track of passive n, initialize as none passive

	g = np.zeros(n) ## need to optimize this  vector
	w = np.dot((B.T),y - B.dot(g)) ## initialize weight
	wE = w * E ## weight of each n
	t = np.argmax(wE) ## find n with max weight
	v = wE[t] ## get max weight value

	while np.sum(E) > 0 and v > eps:

		# make highest weighted n as passive
		E[t] = 0 
		S[t] = 1

		Bs = B[:, S > 0] # get all passive columns
		
		#find beta for all passive columns 
		zsol = lstsq(Bs, y, rcond = -1)[0]
		
		zz = np.zeros(n)
		zz[S > 0] = zsol
		z = zz 

		'''
		inner loop:
		if least square solution has any negative values
		then get the lowest 
		'''
		while np.min(z[S > 0]) <= 0:
			
			# add alpha such that lowest value is zero
			alpha = np.min((g / (g - z))[(S > 0) * (z <= 0)])
			g += alpha * (z - g)

			S[g == 0] = 0 # take out lowest value index from passive
			E[g == 0] = 1 # put back in active 

			Bs = B[:, S > 0] # new passive set
			zsol = lstsq(Bs, y)[0] # calculate least squares soln
			zz = np.zeros(n)
			zz[S > 0] = zsol
			z = zz

		g = z
		w = (B.T).dot(y - B.dot(g))
		wE = w * E
		t = np.argmax(wE)
		v = wE[t]
	return g

def as_solver(B, Y):
	return [np.array([active_set_solver(B, column) for column in Y.T]).T]

def als_activeset(A,k,n_iter,delta=1e-5,tol=0.1):
	
	W = np.random.rand(A.shape[0],k)
	H = np.random.rand(k,A.shape[1])
	loss = []
	for n in range(n_iter):
		print(n)
		H = as_solver(W, A)[0]
		W = as_solver(H.T, A.T)[0].T
		frob_norm = np.linalg.norm(A - np.dot(W,H), 'fro')
		# print("iteration " + str(n + 1) + ": " + str(frob_norm))
		loss.append(frob_norm)
		if frob_norm<tol: break

	return {'W':W,'H':H,'loss':loss}


## non negative lasso method 

def lasso_solver_cwise(B,y):

	from sklearn import linear_model

	clf = linear_model.Lasso(alpha=0.1,positive=True)
	clf.fit(B,y)
	return clf.coef_

def lasso_solver(B, Y):
	return [np.array([lasso_solver_cwise(B, column) for column in Y.T]).T]

def als_lasso(A,k,n_iter,delta=1e-5,tol=0.1):
	
	W = np.random.rand(A.shape[0],k)
	H = np.random.rand(k,A.shape[1])
	loss = []
	for n in range(n_iter):
		print(n)

		H = lasso_solver(W, A)[0]
		W = lasso_solver(H.T, A.T)[0].T

		frob_norm = np.linalg.norm(A - np.dot(W,H), 'fro')
		# print("iteration " + str(n + 1) + ": " + str(frob_norm))
		loss.append(frob_norm)
		if frob_norm<tol: break

	return {'W':W,'H':H,'loss':loss}
