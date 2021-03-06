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

def gauss_logZ(L):
    assert(L.shape[0] == L.shape[1])
    assert(all(np.tril(L_noise) == L_noise))
    return -log(det(L)) + D * log(2 * pi) / 2.

def loglik(X, mu, L):
    D, N = X.shape
    LtXzero = L.T.dot(X - mu.reshape(D, 1))
    l = N * log(det(L)) - D * N * log(2 * pi) / 2.
    l -= np.einsum('ij,ji->', LtXzero.T, LtXzero) / 2.
    return -l


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
    def _init_params(self, D, mu_noise=None, L_noise=None,
                     mu=None, L=None, c=None):
        assert(isinstance(D, np.int))

        mu_noise =  zeros(D) if mu_noise is None else mu_noise
        L_noise =  eye(D) if L_noise is None else L_noise
        assert((np.tril(L_noise) == L_noise).all())
        assert(D == mu_noise.size)
        assert(D == L_noise.shape[0] == L_noise.shape[1])
        self._params_noise = GaussParams(mu_noise, L_noise, 1)

        mu = zeros(D) if mu is None else mu
        L = eye(D) if L is None else L
        c = 1. if c is None else c
        assert((np.tril(L) == L).all())
        assert(D == mu.size)
        assert(D == L.shape[0] == L.shape[1])
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

    def fit_nce(self, X, k=1, mu_noise=None, L_noise=None,
                mu0=None, L0=None, c0=None, method='minimize',
                maxnumlinesearch=None, maxnumfuneval=None, verbose=False):
        _class = self.__class__
        D, Td = X.shape
        self._init_params(D, mu_noise, L_noise, mu0, L0, c0)

        noise = self._params_noise
        Y = mvn.rvs(noise.mu, noise.L, k * Td).T

        maxnumlinesearch = maxnumlinesearch or DEFAULT_MAXNUMLINESEARCH
        obj = lambda u: _class.J(X, Y, noise.mu, noise.L, *vec_to_params(u))
        grad = lambda u: params_to_vec(
            *_class.dJ(X, Y, noise.mu, noise.L, *vec_to_params(u)))

        t0 = params_to_vec(*self._params_nce)
        if method == 'minimize':
            t_star = minimize(t0, obj, grad,
                              maxnumlinesearch=maxnumlinesearch,
                              maxnumfuneval=maxnumfuneval, verbose=verbose)[0]
        else:
            t_star = sp_minimize(obj, t0, method='BFGS', jac=grad,
                                 options={'disp': verbose,
                                          'maxiter': maxnumlinesearch}).x
        self._params_nce = GaussParams(*vec_to_params(t_star))
        return (self._params_nce, Y)

    def fit_ml(self, X):
        D = X.shape[0]
        mu = mean(X, 1)
        L = cholesky(inv(cov(X)))
        c = log(det(L)) - D * log(2 * pi) / 2.
        self._params_ml = GaussParams(mu, L, c)

    @staticmethod
    def _h(U, Uzero, D, k, mu_noise, L_noise, mu, L, c):
        assert(U.shape == Uzero.shape)
        Uzero_noise = U - mu_noise.reshape(D, 1)
        P, P_noise = L.dot(L.T), L_noise.dot(L_noise.T)
        log_pn = log(det(L_noise)) - D * log(2. * pi) / 2.
        log_pn -= np.einsum('ij,jk,ki->i',
                            Uzero_noise.T, P_noise, Uzero_noise) / 2.
        log_pm = -np.einsum('ij,jk,ki->i', Uzero.T, P, Uzero) / 2. + c
        return log_pm - log_pn - log(k)

    @staticmethod
    def J(X, Y, mu_noise, L_noise, mu, L, c):
        """NCE objective function with gaussian data likelihood X and
        gaussian noise Y."""
        assert(mu.size == X.shape[0] == Y.shape[0])
        r = sigmoid
        D, Td = X.shape
        Tn = Y.shape[1]
        k = Tn / Td

        Xzero, Yzero = X - mu.reshape(D, 1), Y - mu.reshape(D, 1)
        h = lambda U, Uzero: NceGauss._h(
            U, Uzero, D, k, mu_noise, L_noise, mu, L, c)
        Jm = -np.sum(log(1 + np.exp(-h(X, Xzero))))
        Jn = -np.sum(log(1 + np.exp(h(Y, Yzero))))

        print("Jm=%10.4f "
              "max(-h(X, Xzero))=%12.3f " %
              (Jm,  max(-h(X, Xzero))))
        print("Jn=%10.4f "
              "max(-h(Y, Yzero))=%12.3f " %
              (Jn,  max(-h(Y, Yzero))))
        print("mu=%s\n; L=%10.4f\n" % (mu, loglik(X, mu, L)))

        return -(Jm + Jn) / Td

    @staticmethod
    def dJ(X, Y, mu_noise, L_noise, mu, L, c):
        """Gradient of the NCE objective function."""
        assert(mu.size == X.shape[0] == Y.shape[0])
        r = sigmoid
        D, Td = X.shape
        Tn = Y.shape[1]
        k = Tn / Td
        P, P_noise = dot(L, L.T), dot(L_noise, L_noise.T)

        Xzero, Yzero = X - mu.reshape(D, 1), Y - mu.reshape(D, 1)
        h = lambda U, Uzero: NceGauss._h(
            U, Uzero, D, k, mu_noise, L_noise, mu, L, c)
        rhX, rhY = r(-h(X, Xzero)), r(h(Y, Yzero))

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

        mu_noise, P_noise = r_[-1., 1.], .5 * c_[[1., .1], [.1, 1.]]
        L_noise = cholesky(P_noise)
        Y = mvn.rvs(mu_noise, inv(P_noise), 200).T

        obj = lambda u: NceGauss.J(
            X, Y, mu_noise, L_noise, *vec_to_params(u))
        grad = lambda u: params_to_vec(
            *NceGauss.dJ(X, Y, mu_noise, L_noise, *vec_to_params(u)))
        grad_diff = lambda u: check_grad(obj, grad, u)

        for i in xrange(100):
            u = r_[0,0,2,0,2,1] + randn(6) / 10
            self.assertLess(grad_diff(u), 1e-5)

    def test_sanity_fit(self):
        mu, P = r_[0., 0.], c_[[2., .2], [.2, 2.]]
        L = cholesky(P)
        Td, k = 100, 2
        X = mvn.rvs(mu, inv(P), Td).T
        theta = GaussParams(zeros(2), eye(2), 1.)

        theta_star, Y = self.model.fit_nce(
            X, k, mu_noise=randn(2), L_noise=(rand() + 1) * eye(2),
            mu0=mu, L0=L, maxnumlinesearch=2000, verbose=False)
        noise = self.model.params_noise
        self.assertLess(NceGauss.J(X, Y, noise.mu, noise.L, *theta_star),
                        NceGauss.J(X, Y, noise.mu, noise.L, *theta))
        self.assertLess(np.sum(params_to_vec(
            *NceGauss.dJ(X, Y, noise.mu, noise.L, *theta_star)) ** 2), 1e-6)


if __name__ == '__main__':
    from scipy.optimize import check_grad
    from scipy.stats import multivariate_normal as mvn

    unittest.main(verbosity=2)
