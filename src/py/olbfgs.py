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
        self._pars_prev = 0

    def _get_initial_direction(self, gradient, stochastic):
        """Determine the approximate product -inv(H) @ grad(F)"""
        p = -gradient
        alphas = []
        # first of 2-loop recursion
        for s, y in zip(reversed(self._s), reversed(self._y)):
            alpha = np.dot(s, p)/np.dot(s, y)
            p -= alpha * y
            alphas.append(alpha)
        if not stochastic:
            if self._iter > 0:
                s, y = self._s[-1], self._y[-1]
                p *= np.dot(s, y)/np.dot(y, y)
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
        self._pars_prev = deepcopy(x)
        self._iter += 1
        # return new parameter vector
        return x + s

    def update_curvature_pairs(self, pars, grad):
        # save curvature pairs
        s = pars - self._pars_prev
        y = grad - self._grad_prev + self._l * s
        self._s.append(s)
        self._y.append(y)


if __name__ == "__main__":
    import scipy.optimize as opt

    # Benchark: optimize an artificially noisy Rosenbrock function
    def noisy_rosenbrock(x, noise_scale):
        #value = opt.rosen(x)
        #gradient = opt.rosen_der(x)
        value = 0
        gradient = np.zeros(x.shape)
        for i in range(0, len(x), 2):
            t1 = 1 - x[i]
            t2 = 10*(x[i+1] - x[i]**2)
            value += t1**2 + t2**2
            gradient[i+1] = 20 * t2
            gradient[i] = -2 * (x[i]*gradient[i+1] + t1)
        value += noise_scale * (np.random.randn() - 0.5)*2 * abs(value)
        gradient += noise_scale * (np.random.randn(len(x)) - 0.5)*2 * max(abs(gradient))
        return value, gradient

    np.random.seed(0)

    # create initial state
    dim = 100
    #x0 = np.random.random(dim)
    x0 = np.ones(dim)
    x0[::2] = -1.2
    x0_gd = deepcopy(x0)
    x0_scipy = deepcopy(x0)

    # step sizes for L-BFGS and gradient descent
    step_size_lbfgs = 0.5
    step_size_gd = 0.001

    # initialize objective function with given noise level
    noise_level = 0.01
    optfun = lambda x: noisy_rosenbrock(x, noise_level)

    # arrays to store function values over the optimization process
    fnvals, fnvals_gd = [], []

    # create L-BFGS object
    hist_size = 5
    lbfgs = LBFGS(hist_size, eps=1e-10)

    value, gradient = optfun(x0)

    xpoints = [deepcopy(x0)]
    xpoints_gd = [deepcopy(x0)]

    # perform optimization
    for it in range(100):
        fnvals.append(value)

        # compute new parameter set
        x0 = lbfgs.iteration(x0, gradient, step_size_lbfgs, stochastic=True)

        # find gradient at new parameter set
        value, gradient = optfun(x0)

        # store curvature pairs
        lbfgs.update_curvature_pairs(x0, gradient)

        # gd for comparison
        value_gd, grad_gd = optfun(x0_gd)
        x0_gd -= step_size_gd * grad_gd
        fnvals_gd.append(value_gd)

        xpoints.append(deepcopy(x0))
        xpoints_gd.append(deepcopy(x0_gd))

        print('LBFGS function value:', value)

    if dim == 2:
        # For 2D Rosenbrock, we can visualize the
        # optimization process
        from mpl_toolkits.mplot3d import Axes3D  
        from matplotlib import cm

        points = np.zeros((len(xpoints), 2))
        points_gd = np.zeros((len(xpoints_gd), 2))
        for i, xp in enumerate(xpoints):
            points[i, :] = xp
            points_gd[i, :] = xpoints_gd[i]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
        y = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
        x, y = np.meshgrid(x, y)
        rosen = lambda x, y: (1 - x**2) + 100*(y - x**2)**2
        z = rosen(x, y)
        ax.plot_surface(x, y, z, cmap=cm.coolwarm)
        ax.plot(points[:, 0], points[:, 1], rosen(points[:, 0], points[:, 1]),
                label="oLBFGS")
        ax.plot(points_gd[:, 0], 
                points_gd[:, 1], 
                rosen(points_gd[:, 0], points_gd[:, 1]),
                label="GD")
        ax.legend()

        plt.show()


    x, f, d = opt.fmin_l_bfgs_b(optfun, x0_scipy)
    print(f"L-BFGS-B minimum: {f}")

    plt.figure()
    plt.semilogy(fnvals, label=fr'O-LBFGS, $\tau={step_size_lbfgs}$, $m={hist_size}$')
    plt.semilogy(fnvals_gd, label=fr'Gradient descent, $\tau={step_size_gd}$')
    plt.legend()
    plt.ylim(min(fnvals + fnvals_gd), max(fnvals + fnvals_gd))
    plt.xlabel('Iteration # (t)')
    plt.ylabel('$F(\mathbf{x}_t)$')
    plt.title("Noisy Rosenbrock function, N = 100")
    plt.show()

