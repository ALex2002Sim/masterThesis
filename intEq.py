import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
from typing import Callable, Dict, Optional, Tuple
import os


from GEI import E3

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

    def __init__(self, d:Optional[Dict[str, float]], func:Callable[[np.float64], np.float64])->None:
        self.L = d['L']
        self.n = d['n']
        self.s = d['s']
        self.kappa = d['kappa']
        self.alpha = self.kappa + self.s
        self.theta_r = d['theta_r']
        self.I_l = d['I_l']
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
        if j == 0:
            res =  0
        elif j == self.n:
            res = self.L
        else:
            res = j*self.h - self.h/2
        
        return res
    
    def buildMatrices(self)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.fill_diagonal(self.A, self.h)
        self.A[0, 0] = self.h/2
        self.A[-1, -1] = self.A[0, 0]


        vals = np.array([self.x(i) for i in np.arange(self.n+1)])
        myEnVals = np.array([E3(self.alpha*(self.L - el)) for el in vals])
        diffVals = np.diff(myEnVals)
        for i in range(self.n):
            for j in range(self.n):
                self.D[i, j] = diffVals[i]*diffVals[j]


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
    
    def solutionGraph(self)->None:
        path = os.path.join('graphs', 'solution.png')
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        self.res = self.solve()
        arr = np.linspace(0, self.L, self.res.size)
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.plot(arr[1:-2], self.res[1:-2], color='fuchsia')
        axs.grid()
        plt.title(f"Solution for $L={self.L}$, $n={self.n}$, $h={self.h:.5f}$, $s={self.s}$, "
                  f"$\\varkappa={self.kappa}$, $\\theta_r={self.theta_r}$, $I_{{\\ell}}={self.I_l}$")
        plt.xlabel(f'$x$', fontsize=12)
        plt.ylabel(f'$\\mathcal{{S}}(x)$', fontsize=12, rotation=0, labelpad=20)
        fig.canvas.manager.set_window_title("Solution")
        
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()

    def errorGraph(self)->None:
        path = os.path.join('graphs', 'error.png')
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        self.res = self.solve()
        arr = np.linspace(0, self.L, self.res.size)[1:-2]
        errors = np.abs(self.res[1:-2] - np.ones(arr.size))
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.plot(arr, errors, color='fuchsia')
        axs.grid()
        plt.title(f"Error for $L={self.L}$, $n={self.n}$, $h={self.h:.5f}$, $s={self.s}$, "
                  f"$\\varkappa={self.kappa}$, $\\theta_r={self.theta_r}$, $I_{{\\ell}}={self.I_l}$")
        plt.xlabel(f'$x$', fontsize=12)
        plt.ylabel(f'$\\varepsilon$', fontsize=12, rotation=0, labelpad=15)
        fig.canvas.manager.set_window_title("Error")
                      

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        