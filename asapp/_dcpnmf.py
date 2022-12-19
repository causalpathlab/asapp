import numpy as np
from scipy import special

class DCPoissonMF():
    def __init__(self, n_components=100, max_iter=50, tol=1e-6,
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
        self.df_a = float(kwargs.get('a', 0.1))
        self.df_b = float(kwargs.get('b', 0.1))

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
        self.c = 1. / np.mean(self.ED)

    def fit(self, X):
        n_samples, n_feats = X.shape
        self._init_frequency(n_feats)
        self._init_depth(n_samples)
        self._update(X)
        return self

    def _update(self, X):
        # alternating between update latent components and weights
        old_bd = -np.inf
        for i in range(self.max_iter):
            self._update_depth(X)
            self._update_frequency(X)
            bound = self._bound(X)
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

    def _update_frequency(self, X):
        self.F_a = self.df_a + np.sum(X, axis=0,keepdims=True)
        self.F_b = self.df_b + np.sum(self.ED, axis=0, keepdims=True)
        self.EF, self.ElogF = _compute_expectations(self.F_a, self.F_b)


    def _update_depth(self, X):
        self.D_a = self.df_a + np.sum(X, axis=1,keepdims=True)
        self.D_b = self.df_b + np.sum(self.EF, axis=1, keepdims=True)
        self.ED, self.ElogD = _compute_expectations(self.D_a, self.D_b)
        self.c = 1. / np.mean(self.ED)

    def _bound(self, X):
        lmbda = np.dot(self.ED,(self.EF))
        bound = np.sum(X * np.log(lmbda) - lmbda)
        return bound

def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return (alpha / beta, special.psi(alpha) - np.log(beta))
