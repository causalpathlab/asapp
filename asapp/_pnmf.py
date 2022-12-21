"""
MODIFIED FROM -->

Poisson matrix factorization with Batch inference and Stochastic inference
CREATED: 2014-03-25 02:06:52 by Dawen Liang <dliang@ee.columbia.edu>
"""

import numpy as np
from scipy import special

class PoissonMF():
    ''' Poisson matrix factorization with batch inference '''
    def __init__(self, n_components=100, max_iter=50, tol=1e-6,
                 smoothness=100, random_state=None, verbose=False,
                 **kwargs):
        ''' Poisson matrix factorization
        Arguments
        ---------
        n_components : int
            Number of latent components
        max_iter : int
            Maximal number of iterations to perform
        tol : float
            The threshold on the increase of the objective to stop the
            iteration
        smoothness : int
            Smoothness on the initialization variational parameters
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
        '''

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
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))

    def _init_components(self, n_feats):
        # variational parameters for beta
        self.gamma_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.rho_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)


    def _init_weights(self, n_samples):
        # variational parameters for theta
        self.gamma_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.rho_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        self.c = 1. / np.mean(self.Et)

    def fit(self, X):
        '''Fit the model to the data in X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
            Training data.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_samples, n_feats = X.shape
        self._init_components(n_feats)
        self._init_weights(n_samples)
        self._update(X)
        return self

    def transform(self, X, attr=None):
        '''Encode the data as a linear combination of the latent components.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
        attr: string
            The name of attribute, default 'Eb'. Can be changed to Elogb to
            obtain E_q[log beta] as transformed data.
        Returns
        -------
        X_new : array-like, shape(n_samples, n_filters)
            Transformed data, as specified by attr.
        '''
        if not hasattr(self, 'Eb'):
            raise ValueError('There are no pre-trained components.')
        n_samples, n_feats = X.shape
        if n_feats != self.Eb.shape[1]:
            raise ValueError('The dimension of the transformed data '
                             'does not match with the existing components.')
        if attr is None:
            attr = 'Et'
        self._init_weights(n_samples)
        self._update(X, update_beta=False)
        return getattr(self, attr)

    def _update(self, X, update_beta=True):
        # alternating between update latent components and weights
        old_bd = -np.inf
        for i in range(self.max_iter):
            self._update_theta(X)
            if update_beta:
                self._update_beta(X)
            bound = self._bound(X)
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

    def _update_theta(self, X):
        ratio = X / self._xexplog()
        self.gamma_t = self.a + np.exp(self.Elogt) * np.dot(
            ratio, np.exp(self.Elogb).T)
        self.rho_t = self.a * self.c + np.sum(self.Eb, axis=1)
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        # print(self.gamma_t.shape,self.rho_t.shape,self.Et.shape, self.Elogt.shape )
        self.c = 1. / np.mean(self.Et)

    def _update_beta(self, X):
        ratio = X / self._xexplog()
        self.gamma_b = self.b + np.exp(self.Elogb) * np.dot(
            np.exp(self.Elogt).T, ratio)
        self.rho_b = self.b + np.sum(self.Et, axis=0, keepdims=True).T
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)
        # print(self.gamma_b.shape,self.rho_b.shape,self.Eb.shape, self.Elogb.shape )

    def _xexplog(self):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogt), np.exp(self.Elogb))

    def _bound(self, X):
        bound = np.sum(X * np.log(self._xexplog()) - self.Et.dot(self.Eb))
        bound += _gamma_term(self.a, self.a * self.c,
                             self.gamma_t, self.rho_t,
                             self.Et, self.Elogt)
        bound += self.n_components * X.shape[0] * self.a * np.log(self.c)
        bound += _gamma_term(self.b, self.b, self.gamma_b, self.rho_b,
                             self.Eb, self.Elogb)
        return bound

def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return (alpha / beta, special.psi(alpha) - np.log(beta))


def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  (special.gammaln(shape) - shape * np.log(rate)))