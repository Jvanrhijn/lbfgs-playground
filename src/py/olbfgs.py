import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import deque
from copy import deepcopy
from functools import reduce

sns.set(style='darkgrid', font_scale=1.5)


class LBFGS:
    """
    Implementation of online L-BFGS (oLBFGS) as per 

        "A Stochastic Quasi-Newton Method for Online Convex Optimization"

		Schraudolf, Yu and Günter (2007)
    
    This implementation skips the 'consistent gradients' step from the
    original paper.
    
    """
    def __init__(self, hist_size, l=0, c=1, eps=1e-10):
        """
        Parameters
	----------
	hist_size: int
		Number of curvature pairs to store internally.
	l: float
		parameter \lambda from the above paper.
	c: float
		parameter c from the above paper.
	eps: float
		parameter \epsilon from the above paper.
	"""
        assert(l >= 0)
        assert(0 < c <= 1)
        self._l = l
        self._c = c
        self._iter = 0
        self._hist_size = hist_size
        self._eps = eps
        self._s = deque(maxlen=hist_size)
        self._y = deque(maxlen=hist_size)
        self._grad_prev = 0

    def _get_initial_direction(self, gradient, stochastic):
        """Determine the approximate product H @ grad(F)"""
        p = -gradient
        alphas = []
        # first of 2-loop recursion
        for s, y in zip(reversed(self._s), reversed(self._y)):
            alpha = np.dot(s, p)/np.dot(s, y)
            p -= alpha * y
            alphas.append(alpha)
        if not stochastic:
            if self._iter > 0:
                p *= np.dot(self._s[-1], self._y[-1])/(self._y[-1]**2).sum()
        else:
            if self._iter == 0:
                p *= self._eps
            else:
                tot = 0
                for (s, y) in zip(self._s, self._y):
                    tot += np.dot(s, y)/np.dot(y, y) 
                p *= tot/min(self._hist_size, self._iter)
        # second of 2-loop recursion
        for alpha, s, y in zip(reversed(alphas), self._s, self._y):
            beta = np.dot(y, p)/np.dot(y, s)
            p += (alpha - beta) * s
        return p


    def iteration(self, x, gradient, step_size=1, stochastic=True):
        """
	Perform a single iteration of oLBFGS.
	
	Parameters
	----------
	x: ndarray
		Current best guess of optimum location.
	step_size: float
		Step size to use for computing curvature parameter s
	stochastic: bool
		Whether to use the stochastically robust version of the algortithm.
		Recommended to leave this on.
	"""
        p = self._get_initial_direction(gradient, stochastic)
        s = step_size * p / self._c
        # save current gradient for next pass
        self._grad_prev = deepcopy(gradient)
        self._iter += 1
        # return new parameter vector
        return x + s

    def update_curvature_pairs(self, s, grad):
        # save curvature pairs
        self._s.append(s)
        self._y.append(grad - self._grad_prev + self._l * s)


if __name__ == "__main__":
    import scipy.optimize as opt

    # Benchark: optimize an artificially noisy Rosenbrock function
    def noisy_rosenbrock(x, noise_scale):
        value = opt.rosen(x)
        value += noise_scale * abs(value) * (np.random.randn() - 0.5)*2
        gradient = opt.rosen_der(x)
        gradient += max(abs(gradient)) * noise_scale * (np.random.randn(len(x)) - 0.5)*2
        return value, gradient

    np.random.seed(0)

    # create initial state
    x0 = np.random.random(100)
    x0_gd = deepcopy(x0)

    # step sizes for L-BFGS and gradient descent
    step_size_lbfgs = 0.5
    step_size_gd = 0.001

    # initialize objective function with given noise level
    noise_level = 0.01
    optfun = lambda x: noisy_rosenbrock(x, noise_level)

    # arrays to store function values over the optimization process
    fnvals, fnvals_gd = [], []

    # create L-BFGS object
    lbfgs = LBFGS(hist_size=10)

    value, gradient = optfun(x0)

    # perform optimization
    for it in range(100):
        fnvals.append(value)
        print('LBFGS function value:', value)

        # compute new parameter set
        xnew = lbfgs.iteration(x0, gradient, step_size_lbfgs, stochastic=True)

        # find gradient at new parameter set
        value, gradient = optfun(xnew)

        # store curvature pairs
        s = xnew - x0
        lbfgs.update_curvature_pairs(s, gradient)

        x0 = xnew

        # gd for comparison
        value_gd, grad_gd = optfun(x0_gd)
        x0_gd -= step_size_gd * grad_gd
        fnvals_gd.append(value_gd)

    plt.figure()
    plt.semilogy(fnvals, label='O-LBFGS')
    plt.semilogy(fnvals_gd, label='Gradient descent')
    plt.legend()
    plt.ylim(min(fnvals + fnvals_gd), max(fnvals + fnvals_gd))
    plt.xlabel('Iteration # (t)')
    plt.ylabel('$F(\mathbf{x}_t)$')
    plt.show()

