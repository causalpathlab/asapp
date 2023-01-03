import numpy as np
from scipy import special

class DCPoissonMF():
    def __init__(self, n_components=10, max_iter=10, tol=1e-3,
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

    def _xexplog(self):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogtheta), np.exp(self.Elogbeta))

    def _update_theta(self, X):
        ratio = X / self._xexplog()        
        self.theta_a = self.t_a + np.exp(self.Elogtheta) * np.dot(ratio, np.exp(self.Elogbeta).T)

        ## base model 
        # self.theta_b = np.tile((self.t_a * self.t_c + np.sum(self.Ebeta, axis=1)).T,self.n_samples).reshape(self.n_samples,self.n_components)

        ## dc model 
        self.theta_b =  self.t_a * self.t_c + np.multiply(np.tile(np.dot(self.Ebeta,self.EF.T),self.n_samples).T,self.ED)
        
        self.Etheta, self.Elogtheta = _compute_expectations(self.theta_a, self.theta_b)
        self.t_c = 1. / np.mean(self.Etheta)

    def _update_beta(self, X):
        ratio = X / self._xexplog()
        self.beta_a = self.b_a + np.exp(self.Elogbeta) * np.dot(np.exp(self.Elogtheta).T, ratio)

        ## base model 
        # self.beta_b = np.tile((self.b_a + np.sum(self.Etheta, axis=0, keepdims=True).T),self.n_feats).reshape(self.n_components,self.n_feats)
        
        ## dc model 
        bb = np.tile(np.sum(np.multiply(self.Etheta,self.ED),axis=0),self.n_feats)
        self.beta_b = self.b_a + np.multiply(bb.reshape(self.n_components,self.n_feats).T,self.EF.T).T

        self.Ebeta, self.Elogbeta = _compute_expectations(self.beta_a, self.beta_b)

    
    def _bound(self, X):
        bound = np.sum(X * np.log(self._xexplog()) - self.Etheta.dot(self.Ebeta))
        bound += _gamma_term(self.t_a, self.t_a * self.t_c,
                             self.theta_a, self.theta_b,
                             self.Etheta, self.Elogtheta)
        bound += self.n_components * X.shape[0] * self.t_a * np.log(self.t_c)
        bound += _gamma_term(self.b_a, self.b_a, self.beta_a, self.beta_b,
                             self.Ebeta, self.Elogbeta)
        return bound
    
    def fit(self, X):
        
        self.n_samples, self.n_feats = X.shape

        self.fit_null(X)

        self._init_beta(self.n_feats)
        self._init_theta(self.n_samples)
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
            self._update_theta(X)
            if update_beta:
                self._update_beta(X)
            bound = self._bound(X)
            # break
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                print('\r\tAfter ITERATION: %d\tObjective: %.2f\t'
                                 'Old objective: %.2f\t'
                                 'Improvement: %.5f' % (i, bound, old_bd,
                                                        improvement))
            if improvement < self.tol:
                break
            old_bd = bound
        if self.verbose:
            print('\n')
        pass

def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return (alpha / beta, special.psi(alpha) - np.log(beta))

def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
        (special.gammaln(shape) - shape * np.log(rate)))

class DCPoissonMFB(DCPoissonMF):
    ''' Poisson matrix factorization with stochastic inference '''
    def __init__(self, n_components=100, batch_size=32, n_pass=10,
                 max_iter=10, tol=0.0005, shuffle=True, smoothness=100,
                 random_state=None, verbose=False,
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

        self._parse_args(**kwargs)
        self._parse_args_batch(**kwargs)

    def _parse_args_batch(self, **kwargs):
        self.t0 = float(kwargs.get('t0', 1.))
        self.kappa = float(kwargs.get('kappa', 0.6))

    def fit(self, X):
        n_samples, n_feats = X.shape
        self._scale = float(n_samples) / self.batch_size
        self._init_beta(n_feats)
        self.bound = list()
        for count in range(self.n_pass):
            if self.verbose:
                print('Iteration %d: passing through the data...' % count)
            indices = np.arange(n_samples)
            if self.shuffle:
                np.random.shuffle(indices)
            X_shuffled = X[indices]
            for (i, istart) in enumerate(range(0, n_samples,self.batch_size), 1):
                print('\tMinibatch %d:' % i)
                iend = min(istart + self.batch_size, n_samples)
                self.set_learning_rate(iter=i)
                mini_batch = X_shuffled[istart: iend]
                self.partial_fit(mini_batch)
                self.bound.append(self._stoch_bound(mini_batch))
        return self

    def partial_fit(self, X):

        self.transform(X)

        ratio = X / self._xexplog()
        
        self.beta_a = (1 - self.rho) * self.beta_a + self.rho * \
            (self.b_a + self._scale * np.exp(self.Elogbeta) * np.dot(np.exp(self.Elogtheta).T, ratio))
        
        bb = np.tile(np.sum(np.multiply(self.Etheta,self.ED),axis=0),self.n_feats)
        self.beta_b = (1 - self.rho) * self.beta_b + self.rho * \
            (self.b_a + self._scale * np.multiply(bb.reshape(self.n_components,self.n_feats).T,self.EF.T).T)

        self.Ebeta, self.Elogbeta = _compute_expectations(self.beta_a, self.beta_b)

        return self

    def set_learning_rate(self, iter=None, rho=None):
        if rho is not None:
            self.rho = rho
        elif iter is not None:
            self.rho = (iter + self.t0)**(-self.kappa)
        else:
            raise ValueError('invalid learning rate.')
        return self

    def _stoch_bound(self, X):
        bound = np.sum(X * np.log(self._xexplog()) - self.Etheta.dot(self.Ebeta))
        bound += _gamma_term(self.t_a, self.t_a * self.t_c,
                             self.theta_a, self.theta_b,
                             self.Etheta, self.Elogtheta)
        bound += self.n_components * X.shape[0] * self.t_a * np.log(self.t_c)
        bound *= self._scale
        bound += _gamma_term(self.b_a, self.b_a, self.beta_a, self.beta_b,
                             self.Ebeta, self.Elogbeta)
        return bound

