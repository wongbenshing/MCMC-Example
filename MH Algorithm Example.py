# Metropolis-Hastings Algorithm (a MCMC method)
# Target distribution p(x) = 0.3*exp(-0.2*x^2) + 0.7*exp(-0.2*(x-10)^2)
# Iteration Times = 5000
# Proposal Distribution: q(x`|x(i)) = N(x(i), sigma=100)

import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt

class MH:
    def __init__(self, exact, N, sigma):
        """
        :param exact: Target distribution
        :param N: Iteration number
        :param sigma: Sigma in Gaussian proposal distribution
        """
        self.p = exact
        self.n = N
        self.q = lambda x: np.random.normal(x, scale=sigma)

    def find_acceptance_probability(self, xi, xs):
        """
        :param xi: x[i]
        :param xs: x[s]
        :return: acceptance probability
        """
        temp = self.p(xs)*self.q(xs) / (self.p(xi)*self.q(xi))
        return min(1, temp)

    def mhAlgorithm(self):
        """
        :return: return the x
        """
        x = [0]  # Initialize x[0] = 0
        for i in range(self.n):
            u = np.random.rand()
            x_star = self.q(x[i])
            ap = self.find_acceptance_probability(x[i], x_star)
            if u < ap:
                x.append(x_star)
            else:
                x.append(x[i])
        return x

    def draw_result(self, x):
        """
        :param x:
        :return: distribution
        """
        # Line for exact function
        xx = np.arange(-10, 20, 0.1)
        yy = np.array([self.p(i) for i in xx])
        plt.plot(xx, yy, '-k')
        # Simulation
        hist = pd.Series(x)
        hist.plot.hist(grid=True, bins=60, range=(-10, 20), color='#607c8e', density=True)
        plt.show()


if __name__ == "__main__":
    myMH = MH(exact=lambda x: 0.3*exp(-0.2*x**2) + 0.7*exp(-0.2*(x-10)**2),
              N=10000, sigma=100)
    res_x = myMH.mhAlgorithm()
    myMH.draw_result(res_x)
