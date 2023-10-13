import numpy as np
import pandas as pd

import scipy.sparse
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler

from ..util._lina import rsvd
from ..dutil.read_write import write_h5


import glob, os
import h5py


def get_bulkdata(bulk_path):
		
	files = []
	for file in glob.glob(bulk_path):
		files.append(file)
	

	dfall = pd.DataFrame()
	cts = []
	for i,f in enumerate(files):
		print('processing...'+str(f))
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
	return dfall,cts

def sim_from_bulk_gamma(bulk_path,sim_data_path,size,alpha,rho,depth,seed):

	np.random.seed(seed)

	df,_ = get_bulkdata(bulk_path)

	nz_cutoff = 10
	df = df[df.iloc[:,2:].sum(1)>nz_cutoff].reset_index(drop=True)
	genes = df['gene'].values
	glens = df['length'].values
	dfbulk = df.drop(columns=['gene','length'])
	beta = np.array(dfbulk.mean(1)).reshape(dfbulk.shape[0],1) 
	# noise = np.array([np.random.gamma(alpha,b/alpha,dfbulk.shape[1]) for b in beta ])
	scale = beta.max()
	noise = np.array([np.random.gamma(scale,1,dfbulk.shape[1]) for x in range(len(beta))])

	dfbulk = (dfbulk * rho) + (1-rho)*noise
	# dfbulk = dfbulk.astype(int)

	## convert to probabilities
	dfbulk = dfbulk.div(dfbulk.sum(axis=0), axis=1)

	all_sc = pd.DataFrame()
	all_indx = []
	ct = {}
	for cell_type in dfbulk.columns:
		sc = pd.DataFrame(np.random.multinomial(depth,dfbulk.loc[:,cell_type],size))
		all_sc = pd.concat([all_sc,sc],axis=0,ignore_index=True)
		all_indx.append([ str(i) + '_' + cell_type.replace(' ','') for i in range(size)])
		if cell_type.split('_')[1] not in ct:
			print(cell_type.split('_')[1])
			ct[cell_type.split('_')[1]] = 1

	dt = h5py.special_dtype(vlen=str) 
	all_indx = np.array(np.array(all_indx).flatten(), dtype=dt) 
	smat = scipy.sparse.csr_matrix(all_sc.values)
	write_h5(sim_data_path,all_indx,genes,smat)

def get_sc(L_total,mu_total,dfct,L_ct,mu_ct,rho,size,depth=1e4):
    
	## sample multivariate normal random variable for total and celltype disributions
	z_total = np.dot(L_total,np.random.normal(size=L_total.shape[1])) + mu_total
	z_ct = np.dot(L_ct,np.random.normal(size=L_ct.shape[1])) + mu_ct
	
	## sample each row(gene) and sort low to high
	x_sample = np.sort(dfct.apply(lambda x: np.random.choice(x), axis=1))
	xz_sample = np.array([np.nan] * len(x_sample))
	
	## combine cell type and total effect
	z = z_ct * np.sqrt(rho) + z_total * np.sqrt(1 - rho)
	
	## get low to high ranking of genes from z and assign
	xz_sample[np.argsort(z)] = x_sample
 
	xz_prop = np.divide(xz_sample, np.sum(xz_sample))
 
	return np.random.multinomial(depth,xz_prop,size)

def simdata_from_bulk_copula(bulk_path,sim_data_path,size,phi,delta,rho,seed):

	np.random.seed(seed)

	df,cts = get_bulkdata(bulk_path)

	nz_cutoff = 10
	df= df[df.iloc[:,2:].sum(1)>nz_cutoff].reset_index(drop=True)
	genes = df['gene'].values
	glens = df['length'].values
	df = df.drop(columns=['gene','length'])

	pnoise = 1 - phi 
	print(rho,phi,delta,pnoise)
 
	x = df.values
	x_log = np.log1p(x)
	x_log_std = (x_log - x_log.mean()) / x_log.std()

	# L = 10
	# U = np.random.normal(size=x.shape[0] * L).reshape(x.shape[0],L)
	# V = np.random.normal(size=x.shape[1] * L).reshape(x.shape[1],L)
	# x_batch = np.dot(U,V.T)
	# x_batch_std = (x_batch - x_batch.mean()) / x_batch.std()

	x_noise = np.random.normal(size=x.shape[0]).reshape(x.shape[0],1)


	x_all = np.exp(x_log_std)*(phi)+ (x_noise*pnoise) 
 
	## normalization of raw data sample wise
	# qt = QuantileTransformer(random_state=0)
	# x_all_q = qt.fit_transform(x_all)

	x_all_q = x_all
	mu_total = x_all_q.mean(1)

	## gene-wise standarization
	scaler = StandardScaler()
	x_all_q_std = scaler.fit_transform(x_all_q.T).T

	## gene-gene correlation using rsvd for mvn input
	u,d,_ = rsvd(x_all_q_std/np.sqrt(x_all_q_std.shape[1]),rank=50)
	L_total = u * d

	dfmix = pd.DataFrame(x_all_q_std,columns=df.columns)
	dfsc = pd.DataFrame()
	all_indx = []
	for ct in cts:
		print('generating single cell data for...'+str(ct))

		dfct = df[[x for x in df.columns if ct in x]]
		dfmix_ct = dfmix[[x for x in dfmix.columns if ct in x]]

		mu_ct = dfmix_ct.mean(1)

		scaler = StandardScaler()
		dfmix_ct = pd.DataFrame(scaler.fit_transform(dfmix_ct.T).T)

		u_ct,d_ct,_ = rsvd(dfmix_ct.to_numpy()/np.sqrt(dfmix_ct.shape[1]),rank=50)
		L_ct = u_ct * d_ct

		ct_sc = get_sc(L_total,mu_total,dfct,L_ct,mu_ct,rho,size)
  
		for i in range(size):
			all_indx.append(str(i) + '_' + ct)

		dfsc = pd.concat([dfsc,pd.DataFrame(ct_sc,columns=genes)],axis=0,ignore_index=True)

	print(dfsc.shape)
	## multiply by genelengths
	smat = scipy.sparse.csr_matrix(dfsc.multiply(glens, axis=1).values)
	dt = h5py.special_dtype(vlen=str) 
	all_indx = np.array(np.array(all_indx).flatten(), dtype=dt) 
	write_h5(sim_data_path,all_indx,genes,smat)

