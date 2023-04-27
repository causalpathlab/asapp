import numpy as np
import pandas as pd

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


def sim_from_bulk_gamma(df,fp,size,alpha,rho,depth,seedn):

	import scipy.sparse
	import scipy.stats as st

	np.random.seed(seedn)

	genes = df['gene'].values
	dfbulk = df.iloc[:,1:] 


	dfbulk = dfbulk.div(dfbulk.sum(axis=0), axis=1)

	dfnoise = dfbulk.copy(deep=True)
	dfnoise = dfnoise.replace(to_replace=dfnoise.values, value=1/dfnoise.shape[0])
	
	all_sc = pd.DataFrame()
	all_indx = []
	for cell_type in dfbulk.columns:
		sc = pd.DataFrame(np.random.multinomial(depth,dfbulk.loc[:,cell_type],size))

		noise = pd.DataFrame(np.random.multinomial(depth,dfnoise.loc[:,cell_type],size))

		sc = (sc * rho) + (1-rho)*noise
		sc = sc.astype(int)

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


def get_sc(L_total,mu_total,dfct,L_ct,mu_ct,rho):
	depth = 10000
	z_total = np.dot(L_total,np.random.normal(size=L_total.shape[1])) + mu_total
	z_ct = np.dot(L_ct,np.random.normal(size=L_ct.shape[1])) + mu_ct
	x_sample = np.sort(dfct.apply(lambda x: np.random.choice(x), axis=1))
	xz_sample = np.array([np.nan] * len(x_sample))
	z = z_ct * np.sqrt(rho) + z_total * np.sqrt(1 - rho)
	xz_sample[np.argsort(z)] = x_sample

	xz_prop = np.divide(xz_sample, np.sum(xz_sample))
	return list(np.random.multinomial(depth,xz_prop,1)[0])



def sim_from_bulk(bulk_path,fp,size,phi,delta,rho,seedn,use_prop=None,ct_prop=None):
	
	import scipy.sparse
	from sklearn.preprocessing import QuantileTransformer
	from sklearn.preprocessing import StandardScaler
	# from sklearn.utils.extmath import randomized_svd as rsvd
	from  asap.util.lina import rsvd
	np.random.seed(seedn)

	import glob, os
	files = []
	for file in glob.glob(bulk_path):
		files.append(file)
	

	dfall = pd.DataFrame()
	cts = []
	for i,f in enumerate(files):

		df = pd.read_csv(f)
		df = df[df['Additional_annotations'].str.contains('protein_coding')].reset_index(drop=True)
		df = df.drop(columns=['Additional_annotations'])
		
		ct = os.path.basename(f).split('.')[0].replace('_TPM','')
		cols = [str(x)+'_'+ct for x in range(df.shape[1]-2)]
		df.columns = ['gene','length'] + cols
		
		if i == 0:
			dfall = df
		else:
			dfall = pd.merge(dfall,df,on=['gene','length'],how='outer')
		cts.append(ct)

	## remove zero genes
	nz_cutoff = 10
	dfall = dfall[dfall.iloc[:,2:].sum(1)>nz_cutoff].reset_index(drop=True)
	genes = dfall['gene'].values
	glens = dfall['length'].values
	dfall = dfall.drop(columns=['gene','length'])

	L = 10
	pnoise = 1 - phi - delta

	x = dfall.values
	x_log = np.log1p(x)
	x_log_std = (x_log - x_log.mean()) / x_log.std()

	U = np.random.normal(size=dfall.shape[0] * L).reshape(dfall.shape[0],L)
	V = np.random.normal(size=dfall.shape[1] * L).reshape(dfall.shape[1],L)
	x_batch = np.dot(U,V.T)
	x_batch_std = (x_batch - x_batch.mean()) / x_batch.std()

	x_noise = np.random.normal(size=dfall.shape[0]).reshape(dfall.shape[0],1)


	x_all = np.exp((x_log_std*np.sqrt(phi)) + (x_batch_std*np.sqrt(delta)) +(x_noise*np.sqrt(pnoise)) )
	dfall = pd.DataFrame(x_all,columns=dfall.columns)

	## normalization of raw data
	qt = QuantileTransformer(random_state=0)
	dfall_q = qt.fit_transform(dfall)

	mu_total = np.mean(dfall_q,axis=1)

	## gene-wise scaling
	scaler = StandardScaler()
	dfall_q = pd.DataFrame(scaler.fit_transform(dfall_q.T).T,columns=dfall.columns)

	## gene-gene correlation using rsvd for mvn input
	u,d,_ = rsvd(dfall_q.to_numpy()/np.sqrt(dfall_q.shape[1]),rank=50)
	L_total = u * d


	dfsc = pd.DataFrame()
	all_indx = []
	for ct in cts:
		dfct = dfall[[x for x in dfall.columns if ct in x]]
		dfct_q = dfall_q[[x for x in dfall_q.columns if ct in x]]

		mu_ct = dfct_q.mean(1)

		scaler = StandardScaler()
		dfct_q = pd.DataFrame(scaler.fit_transform(dfct_q.T).T)

		u,d,_ = rsvd(dfct_q.to_numpy()/np.sqrt(dfct_q.shape[1]),rank=50)
		L_ct = u * d

		ct_sc = []

		if use_prop:
			for i in range(int((ct_prop[ct]/100) * size)):
				ct_sc.append(get_sc(L_total,mu_total,dfct,L_ct,mu_ct,rho))
				all_indx.append(str(i) + '_' + ct)
		else:
			for i in range(size):
				ct_sc.append(get_sc(L_total,mu_total,dfct,L_ct,mu_ct,rho))
				all_indx.append(str(i) + '_' + ct)

		df_ctsc = pd.DataFrame(ct_sc,columns=genes)

		dfsc = pd.concat([dfsc,df_ctsc],axis=0,ignore_index=True)

	## multiply by genelengths
	smat = scipy.sparse.csr_matrix(dfsc.multiply(glens, axis=1).values)

	np.savez(fp,
		indptr = smat.indptr,
		indices = smat.indices,
		data = smat.data)

	dfcols = pd.DataFrame(genes)
	dfcols.columns = ['cols']
	dfcols.to_csv(fp+'.cols.csv.gz',index=False)

	dfrows = pd.DataFrame(np.array(all_indx))
	dfrows.columns = ['rows']
	dfrows.to_csv(fp+'.rows.csv.gz',index=False)
