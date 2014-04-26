#!env python
from __future__ import print_function, division
from collections import namedtuple
import unittest

import numpy as np
from numpy import array, concatenate, dot, eye, log, outer, zeros, \
    r_, c_, pi, mean, cov
from numpy.random import rand, randn
from numpy.linalg import cholesky, det, inv
from scipy import optimize
from scipy.stats import multivariate_normal as mvn

from minimize import minimize

sp_minimize = optimize.minimize


DEFAULT_MAXNUMLINESEARCH = 150

sigmoid = lambda u: 1.0 / (1.0 + np.exp(-u))
GaussParams = namedtuple('GaussParams', ['mu', 'L', 'c'])


def loglik(X, mu, L):
    D, N = X.shape
    Xzero = X - mu.reshape(D, 1)
    P = L.dot(L.T)
    L = -D * N * log(2 * pi) / 2. + N * log(det(P)) / 2.
    L -= np.einsum('ij,ji->', Xzero.T.dot(P), Xzero) / 2.
    return -L


def loglik_vec(X, theta):
    D = X.shape[0]
    (mu, L, c) = vec_to_params(theta)
    return loglik(X, mu, L)


def params_to_vec(mu, L, c):
    assert(mu.size == L.shape[0] == L.shape[1])
    D = mu.size
    c = np.array([c]) if isinstance(c, np.float) else c
    assert(isinstance(c, np.ndarray))
    return concatenate((mu, L[np.tril_indices(D)], c))


def vec_to_params(theta):
    D = (np.sqrt(8. * theta.size + 1) - 3.) / 2.
    assert(int(D) == D)
    D = int(D)
    assert(theta.size == D + D * (D + 1) / 2 + 1)

    mu = theta[0:D]
    L_elems = theta[D:D + D * (D + 1) / 2]
    L = zeros((D, D))
    L[np.tril_indices(D)] = L_elems
    c = theta[-1]
    return GaussParams(mu, L, c)


class NceGauss(object):
    def __init__(self):
        self._D = None

    def _init_params(self, D, mu0=None, P0=None, c0=None):
        assert(isinstance(D, np.int))
        self._D = D
        mu = zeros(D) if mu0 is None else mu0
        P = eye(D) if P0 is None else P0
        c = 1. if c0 is None else c0
        L = cholesky(P)
        assert(D == mu.size)
        assert(D == P.shape[0] == P.shape[1])
        assert(isinstance(c, np.float) or c.size == 1)
        self._params_nce = GaussParams(mu, L, c)

    @property
    def params_nce(self):
        assert(hasattr(self, '_params_nce'))
        return self._params_nce

    @property
    def params_noise(self):
        assert(hasattr(self, '_params_noise'))
        return self._params_noise

    @property
    def params_ml(self):
        assert(hasattr(self, '_params_ml'))
        return self._params_ml


    def fit_nce(self, X, k=1, mu_noise=None, P_noise=None,
                mu0=None, P0=None, c0=None,
                maxnumlinesearch=None, maxnumfuneval=None, verbose=False):
        _class = self.__class__

        D, Td = X.shape
        mu_noise =  zeros(D) if mu_noise is None else mu_noise
        P_noise =  eye(D) if P_noise is None else P_noise
        assert(D == mu_noise.size)
        assert(D == P_noise.shape[0] == P_noise.shape[1])
        Y = mvn.rvs(mu_noise, P_noise, k * Td).T

        self._init_params(D, mu0, P0, c0)
        t0 = params_to_vec(*self._params_nce)

        maxnumlinesearch = maxnumlinesearch or DEFAULT_MAXNUMLINESEARCH
        obj = lambda u: _class.J(X, Y, *vec_to_params(u))
        grad = lambda u: params_to_vec(*_class.dJ(X, Y, *vec_to_params(u)))

        # t_star = minimize(t0, obj, grad, maxnumlinesearch=maxnumlinesearch,
        #                   maxnumfuneval=maxnumfuneval, verbose=verbose)
        t_star = sp_minimize(obj, t0, method='BFGS', jac=grad,
                             options={'disp': verbose,
                                      'maxiter': maxnumlinesearch})
        t_star = (t_star.x, array([1]))

        self._params_nce = GaussParams(*vec_to_params(t_star[0]))
        return (self._params_nce, Y)

    def fit_ml(self, X):
        D = X.shape[0]
        mu = mean(X, 1)
        P = inv(cov(X))
        c = log(det(P)) / 2. - D * log(2 * pi) / 2.
        self._params_ml = GaussParams(mu, P, c)

    @staticmethod
    def _hv(U, Uzero, D, k, mu, P, c):
        assert(U.shape == Uzero.shape)
        log_pn = -D * log(2. * pi) / 2. - np.einsum('ij,ji->i', U.T, U) / 2.
        log_pm = -np.einsum('ij,jk,ki->i', Uzero.T, P, Uzero) / 2. + c
        return log_pm - log_pn - log(k)

    @staticmethod
    def J(X, Y, mu, L, c):
        """NCE objective function with gaussian data likelihood X and
        gaussian noise Y."""
        assert(mu.size == X.shape[0] == Y.shape[0])
        r = sigmoid
        D, Td = X.shape
        Tn = Y.shape[1]
        k = Tn / Td
        P = dot(L, L.T)

        Xzero, Yzero = X - mu.reshape(D, 1), Y - mu.reshape(D, 1)
        hvv = lambda U, Uzero: NceGauss._hv(U, Uzero, D, k, mu, P, c)

        Jm = -np.sum(log(1 + np.exp(-hvv(X, Xzero))))
        Jn = -np.sum(log(1 + np.exp(hvv(Y, Yzero))))
        return -(Jm + Jn) / Td

    @staticmethod
    def dJ(X, Y, mu, L, c):
        """Gradient of the NCE objective function."""
        assert(mu.size == X.shape[0] == Y.shape[0])
        r = sigmoid
        D, Td = X.shape
        Tn = Y.shape[1]
        k = Tn / Td
        P = dot(L, L.T)

        Xzero, Yzero = X - mu.reshape(D, 1), Y - mu.reshape(D, 1)
        hvv = lambda U, Uzero: NceGauss._hv(U, Uzero, D, k, mu, P, c)
        rhX, rhY = r(-hvv(X, Xzero)), r(hvv(Y, Yzero))

        dmu = np.sum(rhX * dot(P, Xzero), 1) - np.sum(rhY * dot(P, Yzero), 1)
        dmu /= Td

        dL = -np.einsum('k,ik,jk->ij', rhX, Xzero, L.T.dot(Xzero))
        dL += np.einsum('k,ik,jk->ij', rhY, Yzero, L.T.dot(Yzero))
        dL /= Td

        dc = (np.sum(rhX) - np.sum(rhY)) / Td
        return (-dmu, -dL, -array([dc]))


class NceGaussTests(unittest.TestCase):
    def setUp(self):
        self.model = NceGauss()

    def test_check_grad(self):
        D = 2
        S = c_[[2., .2], [.2, 2.]]
        X = mvn.rvs(randn(2), S, 100).T
        Y = mvn.rvs(r_[0, 1], eye(2), 200).T

        obj = lambda u: NceGauss.J(X, Y, *vec_to_params(u))
        grad = lambda u: \
               params_to_vec(*NceGauss.dJ(X, Y, *vec_to_params(u)))
        grad_diff = lambda u: check_grad(obj, grad, u)

        for i in xrange(100):
            u = r_[0,0,2,0,2,1] + randn(6) / 10
            self.assertLess(grad_diff(u), 1e-5)

    def test_sanity_fit(self):
        mu = r_[0., 0.]
        S = c_[[2., .2], [.2, 2.]]
        P = np.linalg.inv(S)
        Td, k = 100, 2
        X = mvn.rvs(mu, S, Td).T
        theta = GaussParams(zeros(2), eye(2), 1.)
        theta_star, Y = self.model.fit_nce(
            X, k, mu0=mu, P0=P, maxnumlinesearch=2000, verbose=True)
        self.assertLess(
            NceGauss.J(X, Y, *theta_star), NceGauss.J(X, Y, *theta))
        self.assertLess(
            np.sum(params_to_vec(*NceGauss.dJ(X, Y, *theta_star)) ** 2), 1e-6)


if __name__ == '__main__':
    from scipy.optimize import check_grad
    from scipy.stats import multivariate_normal as mvn

    unittest.main(verbosity=2)
