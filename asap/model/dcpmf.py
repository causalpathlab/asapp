import sys
import numpy as np
from scipy import special
from sklearn.base import BaseEstimator, TransformerMixin
import logging
logger = logging.getLogger(__name__)


class DCNullPoissonMF(BaseEstimator, TransformerMixin):
    def __init__(self,  max_iter=10, tol=1e-6, smoothness=100, random_state=None,**kwargs):
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self._parse_args(**kwargs)
        self._set_random()

    def _set_random(self):
        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)
        else:
            np.random.seed(42)

    def _parse_args(self, **kwargs):
        self.df_a = float(kwargs.get('df_a', 0.1))
        self.df_b = float(kwargs.get('df_b', 0.1))

    def _init_frequency(self, n_feats):
        self.F_a = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(1, n_feats))
        self.F_b = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(1, n_feats))
        self.EF, self.ElogF = self._compute_expectations(self.F_a, self.F_b)

    def _init_depth(self, n_samples):
        self.D_a = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(n_samples, 1))
        self.D_b = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(n_samples,1))
        self.ED, self.ElogD = self._compute_expectations(self.D_a, self.D_b)

    def _update_depth(self, X):
        self.D_a = self.df_a + np.sum(X, axis=1,keepdims=True)
        self.D_b = self.df_b + np.sum(self.EF, axis=1, keepdims=True)
        self.ED, self.ElogD = self._compute_expectations(self.D_a, self.D_b)

    def _update_frequency(self, X):
        self.F_a = self.df_a + np.sum(X, axis=0,keepdims=True)
        self.F_b = self.df_b + np.sum(self.ED, axis=0, keepdims=True)
        self.EF, self.ElogF = self._compute_expectations(self.F_a, self.F_b)

    def _update_null(self, X):
        old_bd = -sys.maxsize - 1
        for _ in range(self.max_iter):
            self._update_depth(X)
            self._update_frequency(X)
            llk = self._llk_null(X)
            improvement = (llk - old_bd) / abs(old_bd)                                                        
            if improvement < self.tol:
                break
            old_bd = llk

    def fit_null(self, X):
        n_samples, n_feats = X.shape
        self._init_frequency(n_feats)
        self._init_depth(n_samples)
        self._update_null(X)
        self._update_baseline()

    def _llk_null(self, X):
        lmbda = np.dot(self.ED,(self.EF))
        llk = np.sum(X * np.log(lmbda) - lmbda)
        return llk

    def _update_baseline(self):
        S = np.sum(self.EF)
        self.EF = self.EF/S
        self.ED = self.ED * S

    def _compute_expectations(self, a, b):
        return (a/b, special.digamma(a) - np.log(b))

class DCPoissonMF(DCNullPoissonMF):
    def __init__(self, n_components=10, max_iter=50, tol=1e-6,
                 smoothness=100, random_state=None,
                 **kwargs):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self._parse_args(**kwargs)
        self._set_random()

    def _parse_args(self, **kwargs):
        self.df_a = float(kwargs.get('df_a', 0.1))
        self.df_b = float(kwargs.get('df_b', 0.1))
        self.t_a = float(kwargs.get('t_a', 0.1))
        self.b_a = float(kwargs.get('b_a', 0.1))

    def _init_beta(self, n_feats):
        self.beta_a = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(self.n_components, n_feats))
        self.beta_b = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(self.n_components, n_feats))
        self.Ebeta, self.Elogbeta = self._compute_expectations(self.beta_a, self.beta_b)

    def _init_theta(self, n_samples):
        self.theta_a = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(n_samples, self.n_components))
        self.theta_b = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(n_samples,self.n_components))
        self.Etheta, self.Elogtheta = self._compute_expectations(self.theta_a, self.theta_b)
        self.t_c = 1. / np.mean(self.Etheta)
    
    def _xexplog(self):
        return np.dot(np.exp(self.Elogtheta), np.exp(self.Elogbeta))

    def _update_xaux(self,X):
        self.xaux = X / self._xexplog()

    def _update_theta(self):
        self.theta_a = self.t_a +  np.exp(self.Elogtheta) * np.dot(self.xaux, np.exp(self.Elogbeta).T)
        bb = np.repeat(np.dot(self.Ebeta,self.EF.T),self.n_samples,axis=1).T
        self.theta_b =  self.t_a * self.t_c + np.multiply(bb,self.ED)
        self.Etheta, self.Elogtheta = self._compute_expectations(self.theta_a, self.theta_b)
        self.t_c = 1. / np.mean(self.Etheta)

    def _update_beta(self):
        self.beta_a = self.b_a +  np.exp(self.Elogbeta) * np.dot(np.exp(self.Elogtheta).T, self.xaux)
        bb = np.repeat(np.sum(np.multiply(self.Etheta,self.ED),axis=0),self.n_feats)
        self.beta_b = self.b_a + np.multiply(bb.reshape(self.n_components,self.n_feats).T,self.EF.T).T
        self.Ebeta, self.Elogbeta = self._compute_expectations(self.beta_a, self.beta_b)
                
    def fit(self, X):
        self.n_samples, self.n_feats = X.shape
        self.fit_null(X)
        self._init_beta(self.n_feats)
        self._init_theta(self.n_samples)
        self._update(X)


    def _update(self, X, update_beta=True):
        self.llk = []
        for _ in range(self.max_iter):
            self._update_xaux(X)
            self._update_theta()
            if update_beta:
                self._update_xaux(X)
                self._update_beta()
            self.llk.append(self._llk(X))
    
    def predict_theta(self, X, predict_iter):
        self.n_samples, self.n_feats = X.shape    
        self.fit_null(X)
        self._init_theta(self.n_samples)
        self._update_xaux(X)
        for _ in range(predict_iter):
            self._update_xaux(X)
            self._update_theta()
        
        return {'theta_a':self.theta_a,
                'theta_b':self.theta_b,
                'depth_a':self.D_a,
                'depth_b':self.D_b,
                'freq_a':self.F_a,
                'freq_b':self.F_b}

    # def _elbo(self,X):
    #     elbo = 0.0
    #     E_aux = np.einsum('ij,ijk->ijk', X, self.aux)
    #     t_b = self.t_a * self.t_c
    #     E_log_pt = self.t_a * np.log(t_b) - special.gammaln(self.t_a) + (self.t_a -1)*self.Elogtheta - t_b*self.Etheta
    #     E_log_pb = self.b_a * np.log(self.b_a) - special.gammaln(self.b_a) + (self.b_a -1)*self.Elogbeta - self.b_a*self.Ebeta
    #     E_log_qt = self.theta_a * np.log(self.theta_b) - special.gammaln(self.theta_a) + (self.theta_a -1)*self.Elogtheta - self.theta_b*self.Etheta
    #     E_log_qb = self.beta_a * np.log(self.beta_a) - special.gammaln(self.beta_a) + (self.beta_a -1)*self.Elogbeta - self.beta_a*self.Ebeta
    #     elbo += np.sum(np.einsum('ijk,ik->i', E_aux,self.Elogtheta)) + np.einsum('ijk,jk->i', E_aux,self.Elogbeta.T)
    #     elbo -= np.einsum('ik,jk->i', self.Etheta,self.Ebeta.T)
    #     elbo += np.sum(E_log_pt)
    #     elbo += np.sum(E_log_pb)
    #     elbo -= np.sum(special.gammaln(X + 1.))
    #     elbo -= np.sum(np.einsum('ijk->i', E_aux * np.log(self.aux)))
    #     elbo -= np.sum(E_log_qt)
    #     elbo -= np.sum(E_log_qb)
    #     return np.mean(elbo)

    def _llk(self,X):
        return np.sum(X * np.log(self._xexplog()) - self.Etheta.dot(self.Ebeta))

class DCPoissonMFSVB(DCPoissonMF):
    def __init__(self, n_components=10, batch_size=32, n_pass=25,
                 max_iter=5 , tol=1e-6, shuffle=True, smoothness=100,
                 random_state=None,verbose=True,
                 **kwargs):

        self.n_components = n_components
        self.batch_size = batch_size
        self.n_pass = n_pass
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        self._parse_args(**kwargs)
        self._parse_args_batch(**kwargs)
        self._set_random()

    def _parse_args_batch(self, **kwargs):
        self.t0 = float(kwargs.get('t0', 1.))
        self.kappa = float(kwargs.get('kappa', 0.2))

    def fit(self, X):
        n_samples, n_feats = X.shape
        self._scale = float(n_samples) / self.batch_size
        self._init_beta(n_feats)
        self.llk = []
        for p in range(self.n_pass):
            # if p%10==0:
            logging.info('Pass over entire data round...'+str(p))
            indices = np.arange(n_samples)
            if self.shuffle:
                np.random.shuffle(indices)
            X_shuffled = X[indices]
            llk = 0.0
            for (i, istart) in enumerate(range(0, n_samples,self.batch_size), 1):
                iend = min(istart + self.batch_size, n_samples)
                self.set_learning_rate(iter=i)
                mini_batch = X_shuffled[istart: iend]
                self.partial_fit(mini_batch)
                llk += self._llk(mini_batch)
            self.llk.append(np.sum(llk))
        return self

    def partial_fit(self, X):
        ## optimize local parameters for minibatch
        self.n_samples, self.n_feats = X.shape
        self.fit_null(X)
        self._init_theta(self.n_samples)
        self._update_xaux(X)
        for i in range(self.max_iter):
            self._update_xaux(X)
            self._update_theta()

        ## update global
        self.beta_a = (1 - self.rho) * self.beta_a + self.rho * \
            (self.b_a + self._scale * np.exp(self.Elogbeta) * np.dot(np.exp(self.Elogtheta).T, self.xaux))
        bb = np.repeat(np.sum(np.multiply(self.Etheta,self.ED),axis=0),self.n_feats)
        self.beta_b = (1 - self.rho) * self.beta_b + self.rho * \
            (self.b_a + self._scale * np.multiply(bb.reshape(self.n_components,self.n_feats).T,self.EF.T).T)
        self.Ebeta, self.Elogbeta = self._compute_expectations(self.beta_a, self.beta_b)

    def set_learning_rate(self, iter):
        self.rho = (iter + self.t0)**(-self.kappa)

class DCPoissonMFMVB(DCPoissonMF):
    def __init__(self, n_components=10, batch_size=32, n_pass=25,
                 max_iter=5 , tol=1e-6, shuffle=True, smoothness=100,
                 random_state=None,verbose=True,
                 **kwargs):

        self.n_components = n_components
        self.batch_size = batch_size
        self.n_pass = n_pass
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        self._parse_args(**kwargs)
        self._set_random()

    def _init_mem_beta(self, n_feats):
        beta_a = np.zeros(shape=(self.n_components, n_feats))
        beta_b = np.zeros(shape=(self.n_components, n_feats))
        return {'a':beta_a,'b':beta_b}

    def initialize_mem(self,X):
        self.all_beta_ab = self._init_mem_beta(self.n_feats)
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        X_shuffled = X[indices]
        self.batch_ids = []
        self.mb_x = {}
        self.mb_beta_ab = {}
        for (i, istart) in enumerate(range(0, self.n_samples,self.batch_size), 1):
            iend = min(istart + self.batch_size, self.n_samples)
            self.batch_ids.append(i)
            self.mb_x[i] = X_shuffled[istart: iend]
            self.mb_beta_ab[i] = self._init_mem_beta(self.n_feats)
    
    def fit(self,X):
        self.n_samples, self.n_feats = X.shape
        self._init_beta(self.n_feats)
        self.initialize_mem(X)
        self.llk = []
        for p in range(self.n_pass):
            if p%10==0:
                logging.info('Pass over entire data round...'+str(p))
            llk = 0.0
            for batch_id in self.batch_ids:
                self.partial_fit(self.mb_x[batch_id],batch_id)
                llk += self._llk(self.mb_x[batch_id])
            self.llk.append(np.sum(llk))
        return self

    def partial_fit(self, X,batch_id):

        ## optimize local parameters for minibatch
        self.n_samples, self.n_feats = X.shape
        self.fit_null(X)
        self._init_theta(self.n_samples)
        self._update_xaux(X)
        for _ in range(self.max_iter):
            self._update_xaux(X)
            self._update_theta()

        ## subtract batch from global
        self.all_beta_ab['a'] -= self.mb_beta_ab[batch_id]['a']
        self.all_beta_ab['b'] -= self.mb_beta_ab[batch_id]['b']

        ## update batch 
        self.mb_beta_ab[batch_id]['a'] = np.exp(self.Elogbeta) * np.dot(np.exp(self.Elogtheta).T, self.xaux)
        bb = np.repeat(np.sum(np.multiply(self.Etheta,self.ED),axis=0),self.n_feats)
        self.mb_beta_ab[batch_id]['b'] = np.multiply(bb.reshape(self.n_components,self.n_feats).T,self.EF.T).T

        ## add batch to global
        self.all_beta_ab['a'] += self.mb_beta_ab[batch_id]['a']
        self.all_beta_ab['b'] += self.mb_beta_ab[batch_id]['b']
        
        ## update global
        self.beta_a = self.b_a + self.all_beta_ab['a']
        self.beta_b = self.b_a + self.all_beta_ab['b']
        self.Ebeta, self.Elogbeta = self._compute_expectations(self.beta_a, self.beta_b)

