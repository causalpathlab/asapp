import numpy as np
from scipy import special

class DCPoissonMF():
    def __init__(self, n_components=10, max_iter=10, tol=1e-6,
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

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.df_a = float(kwargs.get('df_a', 0.1))
        self.df_b = float(kwargs.get('df_b', 0.1))
        self.t_a = float(kwargs.get('t_a', 0.1))
        self.b_a = float(kwargs.get('b_a', 0.1))

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
        print('updaing null model....')
        old_bd = -np.inf
        for i in range(self.max_iter):
            self._update_depth(X)
            self._update_frequency(X)
            bound = self._bound_null(X)
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                print('\r\tAfter ITERATION: %d\tObjective: %.2f\t'
                                 'Old objective: %.2f\t'
                                 'Improvement: %.5f' % (i, bound, old_bd,
                                                        improvement))
                                                        
                frob_norm = np.linalg.norm(X - np.dot(self.ED,self.EF), 'fro')
                print(frob_norm)
            if improvement < self.tol:
                break
            old_bd = bound
        if self.verbose:
            print('\n')
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

    ### model
    def _update_aux(self,esp=1e-7):
        
        #### using digamma 
        theta = special.psi(self.theta_a) - np.log(self.theta_b)
        beta = special.psi(self.beta_a) - np.log(self.beta_b)
        aux = np.einsum('ik,jk->ijk', np.exp(theta), np.exp(beta).T) + esp
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
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogtheta), np.exp(self.Elogbeta))

    def _update_theta(self, X):

        self.theta_a = self.t_a + np.einsum('ij,ijk->ik',X,self.aux)
        ## dc model 
        # self.theta_b =  self.t_a * self.t_c + self.ED * np.einsum('j,jk->ik',self.EF,self.Ebeta,),self.n_samples).T,

        # self.theta_a = self.t_a + np.dot(X,self.aux.sum(axis=0))
        # ## dc model 
        self.theta_b =  self.t_a * self.t_c + np.multiply(np.tile(np.dot(self.Ebeta,self.EF.T),self.n_samples).T,self.ED)
        
        self.Etheta, self.Elogtheta = _compute_expectations(self.theta_a, self.theta_b)
        self.t_c = 1. / np.mean(self.Etheta)

    def _update_beta(self, X):
        self.beta_a = self.b_a + np.einsum('ij,ijk->kj',X,self.aux)
        ## dc model 
        bb = np.tile(np.sum(np.multiply(self.Etheta,self.ED),axis=0),self.n_feats)
        self.beta_b = self.b_a + np.multiply(bb.reshape(self.n_components,self.n_feats).T,self.EF.T).T

        # self.beta_a = self.b_a + np.dot(self.aux.sum(axis=1).T,X)
        # ## dc model 
        # bb = np.tile(np.sum(np.multiply(self.Etheta,self.ED),axis=0),self.n_feats)
        # self.beta_b = self.b_a + np.multiply(bb.reshape(self.n_components,self.n_feats).T,self.EF.T).T

        self.Ebeta, self.Elogbeta = _compute_expectations(self.beta_a, self.beta_b)

                
    def fit(self, X):
        self.n_samples, self.n_feats = X.shape
        self.fit_null(X)
        self._init_beta(self.n_feats)
        self._init_theta(self.n_samples)
        self._init_aux()
        print(self.theta_a.shape,self.theta_b.shape,self.Etheta.shape,self.Elogtheta.shape)
        print(self.beta_a.shape,self.beta_b.shape,self.Ebeta.shape,self.Elogbeta.shape)
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
        self._update(X, update_beta=False)
        return getattr(self, attr)


    def _update(self, X, update_beta=True):
        old_bd = -np.inf
        for i in range(self.max_iter):
            self._update_aux()
            self._update_theta(X)
            if update_beta:
                self._update_aux()
                self._update_beta(X)
            # bound = self._bound(X)
            bound = self._elbo(X)

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
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return (alpha / beta, special.psi(alpha) - np.log(beta))


