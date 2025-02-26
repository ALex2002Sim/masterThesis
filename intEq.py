import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz
from typing import Callable, Tuple

from GEI import E3
from tests import params

class IntEq:
    """
        Class for solving the integral equation of radiative transfer.

        This class implements a numerical solution for the radiative transfer equation in a layered medium.

        Parameters
        ----------
        L : float
            Thickness of the layer.
        n : int
            Number of partition segments in the interval [0, L].
        s : float
            Dissipation coefficient, representing scattering and absorption losses.
        kappa : float
            Absorption coefficient, quantifying the medium's capacity to absorb radiation.
        theta_r : float
            Refractive index of the bottom boundary of the layer.
        I_l : float
            Intensity of incident radiation at the top boundary.
    """

    def __init__(self, d:params, func:Callable[[np.float64], np.float64])->None:
        self.L = d.L
        self.n = d.n
        self.s = d.s
        self.kappa = d.kappa
        self.alpha = self.kappa + self.s
        self.theta_r = d.theta_r
        self.I_l = d.I_l
        self.h = self.L/self.n

        self.A = np.zeros((self.n, self.n))
        self.E = np.zeros((self.n, self.n))
        self.D = np.zeros((self.n, self.n))
        self.e = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.func = func
        self.f = np.zeros(self.n)

        self.res = np.zeros(self.n)

    def x(self, j:np.int64)->np.float64:
        return np.float64(0 if j == 0 else self.L if j == self.n else j*self.h - self.h/2)

    
    def buildMatrices(self)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.fill_diagonal(self.A, self.h)
        self.A[0, 0] = self.h/2
        self.A[-1, -1] = self.A[0, 0]


        vals = np.array([self.x(i) for i in np.arange(self.n + 1)])
        myEnVals = np.array([E3(self.alpha*(self.L - el)) for el in vals])
        diffVals = np.diff(myEnVals)
        self.D = np.outer(diffVals, diffVals)


        c = np.power(self.alpha, -2)
        a = self.alpha*self.h
        b = self.h*np.power(self.alpha, -1)
        ind = np.arange(1, self.n)
        arr = np.zeros(self.n)
        for i in ind[1:]:
            arr[i-1] = E3(a*(i)) + E3(a*(i-2)) - 2*E3(a*(i-1))
        arr *= c
        arr[0] = 2*b + 2*c*E3(a) - c
        self.E = toeplitz(arr)
        arr = np.zeros(self.n)
        for i in ind[1:]:
            arr[i-1] = -E3(a*(i-1)) + E3(a*(i-2)) + E3(a*(i-0.5)) - E3(a*(i-1.5))
        self.E[:, 0] = c*arr
        arr = np.zeros(self.n)
        for i in ind[:-1]:
            arr[i-1] = -E3(self.alpha*(self.L + self.h*(0.5-i))) + E3(self.alpha*(self.L + self.h*(1.5-i))) + \
                        E3(self.alpha*(self.L-i*self.h)) - E3(self.alpha*(self.L-self.h*(i-1)))
        self.E[:, -1] = c*arr
        self.E[0, 0] = b + 2*c*E3(a/2) - c
        self.E[-1, -1] = self.E[0, 0]

        return self.A, np.power(self.alpha, -2)*self.D, self.E
    
    def buildVectors(self)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for i in range(1, self.n+1):
            self.e[i-1] = E3(self.alpha*self.x(i)) - E3(self.alpha*self.x(i-1))

        for i in range(1, self.n+1):
            self.b[i-1] = E3(self.alpha*(self.L - self.x(i))) - E3(self.alpha*(self.L - self.x(i-1)))

        for i in range(1, self.n+1):
            self.f[i-1] = quad(self.func, self.x(i-1), self.x(i))[0]

        return -np.power(self.alpha, -1)*self.e, np.power(self.alpha, -1)*self.b, self.f

    
    def solve(self)->np.ndarray:
        self.A, self.D, self.E = self.buildMatrices()
        self.e, self.b, self.f = self.buildVectors()

        B = self.A - self.s*self.theta_r*self.D - self.s/2*self.E
        k = self.I_l/2*self.e + self.I_l*self.theta_r*E3(self.alpha*self.L)*self.b + self.f

        self.res = np.linalg.solve(B, k)

        return self.res