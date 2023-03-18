import numpy as np
import pandas as pd
np.random.seed(42)

def sample_gamma(shape, rate, size):
	return np.random.gamma(shape, 1./rate, size=size)

def generate_H(N, K, alpha=1., eps=2.):
	H = sample_gamma(alpha, 1., size=(N,K))
	for k in range(K):
		size = H[k:k+1, int(k*N/K):int((k+1)*N/K)].shape
		H[k:k+1, int(k*N/K):int((k+1)*N/K)] =  sample_gamma(np.random.random(1)/10, 1./1000., size=size)
		H[k:k+1, int(k*N/K):int((k+1)*N/K)] = sample_gamma(alpha + eps, 1./eps, size=size)
	return H

def generate_W(P, K, noise_prop=0., beta=2., eps=4.):

	W = np.zeros((P, K))

    ## add noise
	P_0 = int((1. - noise_prop) * P)
	if noise_prop > 0.:
		size = W[(P-P_0):, :].shape
		W[(P-P_0):, :] = sample_gamma(0.7, 1., size=size)
	W[:P_0, :] = sample_gamma(beta, 1, size=(P_0, K))

	for k in range(K):
		size = W[int(k*P_0/K):int((k+1)*P_0/K), k:k+1].shape
		W[int(k*P_0/K):int((k+1)*P_0/K), k:k+1] = sample_gamma(np.random.random(1)/10, 1./1000., size=size)	
		W[int(k*P_0/K):int((k+1)*P_0/K), k:k+1] = sample_gamma(beta +eps, 1./eps, size=size)	
	
	return W

def sim_from_bulk(df,fp,size,alpha,beta,rho,depth):

	import scipy.sparse 

	genes = df['gene'].values
	dfbulk = df.iloc[:,1:] 
	
	noise = np.random.gamma(alpha, beta,dfbulk.shape[0]*dfbulk.shape[1]).reshape(dfbulk.shape[0],dfbulk.shape[1])
	noise = noise * np.array(dfbulk.mean(1)).reshape(dfbulk.shape[0],1) 
	dfbulk = (dfbulk * rho) + (1-rho)*noise
	dfbulk = dfbulk.astype(int)
	dfbulk = dfbulk.div(dfbulk.sum(axis=0), axis=1)

	all_sc = pd.DataFrame()
	all_indx = []
	for cell_type in dfbulk.columns:
		sc = pd.DataFrame(np.random.multinomial(depth,dfbulk.loc[:,cell_type],size))
		all_sc = pd.concat([all_sc,sc],axis=0,ignore_index=True)
		all_indx.append([ str(i) + '_' + cell_type.replace(' ','') for i in range(size)])
	
	smat = scipy.sparse.csr_matrix(all_sc.values)

	np.savez(fp,
        indptr = smat.indptr,
        indices = smat.indices,
        data = smat.data)

	dfcols = pd.DataFrame(genes)
	dfcols.columns = ['cols']
	dfcols.to_csv(fp+'.cols.csv.gz',index=False)

	dfrows = pd.DataFrame(np.array(all_indx).flatten())
	dfrows.columns = ['rows']
	dfrows.to_csv(fp+'.rows.csv.gz',index=False)