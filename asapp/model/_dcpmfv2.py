

import numpy as np
from scipy import special
import logging
logger = logging.getLogger(__name__)


class DCNullPoissonMF():
    def __init__(self,  max_iter=25, tol=1e-6, smoothness=100, random_state=None,verbose=False,**kwargs):

        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)
        else:
            np.random.seed(42)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.df_a = float(kwargs.get('df_a', 0.1))
        self.df_b = float(kwargs.get('df_b', 0.1))

    def _init_frequency(self, n_feats):
        self.F_a = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(1, n_feats))
        self.F_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(1, n_feats))
        self.EF, self.ElogF = _compute_expectations(self.F_a, self.F_b)

    def _init_depth(self, n_samples):
        self.D_a = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(n_samples, 1))
        self.D_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(n_samples,1))
        self.ED, self.ElogD = _compute_expectations(self.D_a, self.D_b)

    def _update_depth(self, X):
        self.D_a = self.df_a + np.sum(X, axis=1,keepdims=True)
        self.D_b = self.df_b + np.sum(self.EF, axis=1, keepdims=True)
        self.ED, self.ElogD = _compute_expectations(self.D_a, self.D_b)
        self.c = 1. / np.mean(self.ED)

    def _update_frequency(self, X):
        self.F_a = self.df_a + np.sum(X, axis=0,keepdims=True)
        self.F_b = self.df_b + np.sum(self.ED, axis=0, keepdims=True)
        self.EF, self.ElogF = _compute_expectations(self.F_a, self.F_b)

    def _update_null(self, X):
        old_bd = -np.inf
        for _ in range(self.max_iter):
            self._update_depth(X)
            self._update_frequency(X)
            bound = self._bound_null(X)
            improvement = (bound - old_bd) / abs(old_bd)                                                        
            if improvement < self.tol:
                break
            old_bd = bound
        pass

    def fit_null(self, X):
        n_samples, n_feats = X.shape
        self._init_frequency(n_feats)
        self._init_depth(n_samples)
        self._update_null(X)
        self._update_baseline()
        return self

    def _bound_null(self, X):
        lmbda = np.dot(self.ED,(self.EF))
        bound = np.sum(X * np.log(lmbda) - lmbda)
        return bound

    def _update_baseline(self):
        S = np.sum(self.EF)
        self.EF = self.EF/S
        self.ED = self.ED * S

class DCPoissonMF(DCNullPoissonMF):
    def __init__(self, n_components=10, max_iter=50, tol=1e-6,
                 smoothness=100, random_state=None, verbose=True,
                 **kwargs):

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)
        else:
            np.random.seed(42)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.df_a = float(kwargs.get('df_a', 0.1))
        self.df_b = float(kwargs.get('df_b', 0.1))
        self.t_a = float(kwargs.get('t_a', 0.1))
        self.b_a = float(kwargs.get('b_a', 0.1))

    def _init_beta(self, n_feats):
        self.beta_a = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(self.n_components, n_feats))
        self.beta_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(self.n_components, n_feats))
        self.Ebeta, self.Elogbeta = _compute_expectations(self.beta_a, self.beta_b)

    def _init_theta(self, n_samples):
        self.theta_a = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(n_samples, self.n_components))
        self.theta_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,size=(n_samples,self.n_components))
        self.Etheta, self.Elogtheta = _compute_expectations(self.theta_a, self.theta_b)
        self.t_c = 1. / np.mean(self.Etheta)
    
    def _init_aux(self):
        self.aux = np.ones((self.theta_a.shape[1],self.theta_a.shape[0],self.beta_a.shape[1]))

    ### model
    def _update_aux(self,eps=1e-7):
        
        #### using digamma 
        theta = special.psi(self.theta_a) - np.log(self.theta_b)
        beta = special.psi(self.beta_a) - np.log(self.beta_b)
        aux = np.einsum('ik,jk->ijk', np.exp(theta), np.exp(beta).T) + eps
        self.aux = aux / (np.sum(aux, axis=2)[:, :, np.newaxis])

        #### using precalculated explog
        # aux = np.zeros((self.theta_a.shape[1],self.theta_a.shape[0],self.beta_a.shape[1]))
        # for i in range(self.aux.shape[0]):
        #     theta = self.Elogtheta[:,i].reshape(self.Elogtheta.shape[0],-1)
        #     beta = self.Elogbeta[i,:].reshape(self.Elogbeta.shape[1],-1).T
        #     aux[i,:,:] = np.dot(np.exp(theta),np.exp(beta)) + 1e-7
        # k_sum = aux.sum(axis=0)
        # for i in range(self.aux.shape[0]):
        #     self.aux[i,:,:] = aux[i,:,:]/k_sum  



    def _xexplog(self):
        return np.dot(np.exp(self.Elogtheta), np.exp(self.Elogbeta))

    def _update_theta(self, X):
        self.theta_a = self.t_a + np.einsum('ij,ijk->ik',X,self.aux)
        self.theta_b =  self.t_a * self.t_c + np.multiply(np.tile(np.dot(self.Ebeta,self.EF.T),self.n_samples).T,self.ED)
        self.Etheta, self.Elogtheta = _compute_expectations(self.theta_a, self.theta_b)
        self.t_c = 1. / np.mean(self.Etheta)

    def _update_beta(self, X):
        self.beta_a = self.b_a + np.einsum('ij,ijk->kj',X,self.aux)
        bb = np.tile(np.sum(np.multiply(self.Etheta,self.ED),axis=0),self.n_feats)
        self.beta_b = self.b_a + np.multiply(bb.reshape(self.n_components,self.n_feats).T,self.EF.T).T
        self.Ebeta, self.Elogbeta = _compute_expectations(self.beta_a, self.beta_b)

                
    def fit(self, X):
        self.n_samples, self.n_feats = X.shape
        self.fit_null(X)
        self._init_beta(self.n_feats)
        self._init_theta(self.n_samples)
        self._init_aux()
        self._update(X)
        return self

    def transform(self, X, attr=None):
        self.n_samples, self.n_feats = X.shape    
        if not hasattr(self, 'Ebeta'):
            raise ValueError('There are no pre-trained components.')
        if self.n_feats != self.Ebeta.shape[1]:
            raise ValueError('The dimension of the transformed data '
                             'does not match with the existing components.')
        if attr is None:
            attr = 'Etheta'

        self.fit_null(X)
        self._init_theta(self.n_samples)
        
        # self._update(X, update_beta=False)
        old_bd = -np.inf
        self.bound_t = []
        for i in range(self.max_iter):
            self._update_aux()
            self._update_theta(X)
            # bound = self._bound(X)
            bound = self._elbo(X)
            self.bound_t.append(bound)
            # break
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                print('\r\tAfter ITERATION: %d\tObjective: %.2f\t'
                                 'Old objective: %.2f\t'
                                 'Improvement: %.5f' % (i, bound, old_bd,
                                                        improvement))
            # if improvement < self.tol:
            #     break
            old_bd = bound
        if self.verbose:
            print('\n')
        pass



    def _update(self, X, update_beta=True):
        old_bd = -np.inf
        self.bound = []
        for i in range(self.max_iter):
            self._update_aux()
            self._update_theta(X)
            if update_beta:
                self._update_aux()
                self._update_beta(X)
            # bound = self._bound(X)
            bound = self._elbo(X)
            self.bound.append(bound)
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                print('\r\tAfter ITERATION: %d\tObjective: %.2f\t'
                                 'Old objective: %.2f\t'
                                 'Improvement: %.5f' % (i, bound, old_bd,
                                                        improvement))
            # if improvement < self.tol:
            #     break
            old_bd = bound
        if self.verbose:
            print('\n')
        pass
    

    def _elbo(self,X):

        elbo = 0.0
        E_aux = np.einsum('ij,ijk->ijk', X, self.aux)
        t_b = self.t_a * self.t_c

        E_log_pt = self.t_a * np.log(t_b) - special.gammaln(self.t_a) + (self.t_a -1)*self.Elogtheta - t_b*self.Etheta
        E_log_pb = self.b_a * np.log(self.b_a) - special.gammaln(self.b_a) + (self.b_a -1)*self.Elogbeta - self.b_a*self.Ebeta

        E_log_qt = self.theta_a * np.log(self.theta_b) - special.gammaln(self.theta_a) + (self.theta_a -1)*self.Elogtheta - self.theta_b*self.Etheta
        E_log_qb = self.beta_a * np.log(self.beta_a) - special.gammaln(self.beta_a) + (self.beta_a -1)*self.Elogbeta - self.beta_a*self.Ebeta

        elbo += np.sum(np.einsum('ijk,ik->i', E_aux,self.Elogtheta)) + np.einsum('ijk,jk->i', E_aux,self.Elogbeta.T)
        elbo -= np.einsum('ik,jk->i', self.Etheta,self.Ebeta.T)
        elbo += np.sum(E_log_pt)
        elbo += np.sum(E_log_pb)
        elbo -= np.sum(special.gammaln(X + 1.))
        elbo -= np.sum(np.einsum('ijk->i', E_aux * np.log(self.aux)))
        elbo -= np.sum(E_log_qt)
        elbo -= np.sum(E_log_qb)
        
        return np.mean(elbo)

def _compute_expectations(alpha, beta):
    return (alpha / beta, special.psi(alpha) - np.log(beta))

 ##  method: Stochastic online variational inference
class DCPoissonMFBatch(DCPoissonMF):
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

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)
        else:
            np.random.seed(42)

        self._parse_args(**kwargs)
        self._parse_args_batch(**kwargs)

    def _parse_args_batch(self, **kwargs):
        self.t0 = float(kwargs.get('t0', 1.))
        self.kappa = float(kwargs.get('kappa', 0.2))

    def fit(self, X):
        n_samples, n_feats = X.shape
        self._scale = float(n_samples) / self.batch_size
        self._init_beta(n_feats)
        self.bound = []
        for count in range(self.n_pass):
            if count%10==0:
                logging.info('Pass over entire data round...'+str(count))
            indices = np.arange(n_samples)
            if self.shuffle:
                np.random.shuffle(indices)
            X_shuffled = X[indices]
            bound = []
            for (i, istart) in enumerate(range(0, n_samples,self.batch_size), 1):
                iend = min(istart + self.batch_size, n_samples)
                self.set_learning_rate(iter=i)
                mini_batch = X_shuffled[istart: iend]
                self.partial_fit(mini_batch)
                bound.append(self._elbo(mini_batch))
            
            self.bound.append(np.mean(bound))
        return self

    def partial_fit(self, X):

        ## optimize local parameters for minibatch
        self.n_samples, self.n_feats = X.shape
        self.fit_null(X)
        self._init_theta(self.n_samples)
        self._init_aux()
        for i in range(self.max_iter):
            self._update_aux()
            self._update_theta(X)

        ## update global
        self.beta_a = (1 - self.rho) * self.beta_a + self.rho * \
            (self.b_a + self._scale * np.einsum('ij,ijk->kj',X,self.aux))
        bb = np.tile(np.sum(np.multiply(self.Etheta,self.ED),axis=0),self.n_feats)
        self.beta_b = (1 - self.rho) * self.beta_b + self.rho * \
            (self.b_a + self._scale * np.multiply(bb.reshape(self.n_components,self.n_feats).T,self.EF.T).T)
        self.Ebeta, self.Elogbeta = _compute_expectations(self.beta_a, self.beta_b)


    def set_learning_rate(self, iter):
        self.rho = (iter + self.t0)**(-self.kappa)


    def _elbo(self,X):

        elbo = 0.0
        E_aux = np.einsum('ij,ijk->ijk', X, self.aux)
        t_b = self.t_a * self.t_c

        E_log_pt = self.t_a * np.log(t_b) - special.gammaln(self.t_a) + (self.t_a -1)*self.Elogtheta - t_b*self.Etheta
        E_log_pb = self.b_a * np.log(self.b_a) - special.gammaln(self.b_a) + (self.b_a -1)*self.Elogbeta - self.b_a*self.Ebeta

        E_log_qt = self.theta_a * np.log(self.theta_b) - special.gammaln(self.theta_a) + (self.theta_a -1)*self.Elogtheta - self.theta_b*self.Etheta
        E_log_qb = self.beta_a * np.log(self.beta_a) - special.gammaln(self.beta_a) + (self.beta_a -1)*self.Elogbeta - self.beta_a*self.Ebeta

        elbo += np.sum(np.einsum('ijk,ik->i', E_aux,self.Elogtheta)) + np.einsum('ijk,jk->i', E_aux,self.Elogbeta.T)
        elbo -= np.einsum('ik,jk->i', self.Etheta,self.Ebeta.T)
        elbo += np.sum(E_log_pt)
        elbo += np.sum(E_log_pb)
        elbo -= np.sum(special.gammaln(X + 1.))
        elbo -= np.sum(np.einsum('ijk->i', E_aux * np.log(self.aux)))
        elbo -= np.sum(E_log_qt)
        elbo -= np.sum(E_log_qb)
        
        return np.mean(elbo)


 ##  method: Memoized online variational inference
