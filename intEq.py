import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
from typing import Callable
import mplcursors
import os

from GEI import E2, E3

class IntEq:
    """
    class that implements the solution to the integral equation of radiative transfer

    Parameters
    ----------
    L: float
        layer thickness
    n: int
        number of partition segments of [0, L]
    s: float
        dissipation coefficient
    kappa: float
        absorption coefficient
    theta_r: float
        refractive index of the bottom part of the layer
    I_l: float
        incident radiation intensity
    """

    def __init__(self, d:dict, func:Callable[[np.float64], np.float64]):
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

    def buildMatrixA(self)->np.ndarray:
        np.fill_diagonal(self.A, self.h)
        self.A[0, 0] = self.h/2
        self.A[-1, -1] = self.A[0, 0]

        return self.A
    
    def buildMatrixD(self)->np.ndarray:
        vals = np.array([self.x(i) for i in np.arange(self.n+1)])
        myEnVals = np.array([E3(self.alpha*(self.L - el)) for el in vals])
        diffVals = np.diff(myEnVals)

        for i in range(self.n):
            for j in range(self.n):
                self.D[i, j] = diffVals[i]*diffVals[j]

        return np.power(self.alpha, -2)*self.D
    
    def buildMatrixE(self)->np.ndarray:
        c:np.float64   = np.power(self.alpha, -2)
        a:np.float64   = self.alpha*self.h
        b:np.float64   = self.h*np.power(self.alpha, -1)
        ind:np.ndarray = np.arange(1, self.n)

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
        
        return self.E
    
    def buildVectorE(self)->np.ndarray:
        for i in range(1, self.n+1):
            self.e[i-1] = E3(self.alpha*self.x(i)) - E3(self.alpha*self.x(i-1))

        return -np.power(self.alpha, -1)*self.e
    
    def buildVectorB(self)->np.ndarray:
        for i in range(1, self.n+1):
            self.b[i-1] = E3(self.alpha*(self.L - self.x(i))) - E3(self.alpha*(self.L - self.x(i-1)))

        return np.power(self.alpha, -1)*self.b
    
    def buildVectorF(self)->np.ndarray:
        for i in range(1, self.n+1):
            self.f[i-1] = quad(self.func, self.x(i-1), self.x(i))[0]

        return self.f
    
    def solve(self)->np.ndarray:
        self.D = self.buildMatrixD()
        self.A = self.buildMatrixA()
        self.E = self.buildMatrixE()
        self.b = self.buildVectorB()
        self.e = self.buildVectorE()
        self.f = self.buildVectorF()

        B = self.A - self.s*self.theta_r*self.D - self.s/2*self.E
        k = self.I_l/2*self.e + self.I_l*self.theta_r*E3(self.alpha*self.L)*self.b + self.f

        self.res = np.linalg.solve(B, k)

        return self.res
    
    def solutionGraph(self, filename:str='solution')->None:
        path = os.path.join('graphs', f'{filename}.png')
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        self.res = self.solve()
        arr = np.linspace(0, self.L, self.res.size)
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        fig.patch.set_facecolor('black')
        axs.set_facecolor('black')
        line, = axs.plot(arr[1:-2], self.res[1:-2], color='fuchsia')
        axs.grid()
        plt.title(f"Solution for $L={self.L}$, $n={self.n}$, $h={self.h:.5f}$, $s={self.s}$, "
                  f"$\\varkappa={self.kappa}$, $\\theta_r={self.theta_r}$, $I_{{\\ell}}={self.I_l}$", color='white')
        plt.xlabel(f'$x$', fontsize=12, color='white')
        plt.ylabel(f'$\\mathcal{{S}}(x)$', fontsize=12, rotation=0, labelpad=20, color='white')
        axs.tick_params(colors='white')
        fig.canvas.manager.set_window_title("Solution")

        cursor = mplcursors.cursor(line, hover=True)
        cursor.connect("add",
                        lambda sel: (
                               sel.annotation.set_text(f"$x$={sel.target[0]:.3f}\n$\\mathcal{{S}}$={sel.target[1]:.3f}"),
                               sel.annotation.set_color('white'),
                               sel.annotation.get_bbox_patch().set_facecolor("violet"),
                               sel.annotation.get_bbox_patch().set_edgecolor("purple"),))
        
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()

    def errorGraph(self, filename:str='error')->None:
        path = os.path.join('graphs', f'{filename}.png')
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        self.res = self.solve()
        arr = np.linspace(0, self.L, self.res.size)[1:-2]
        errors = np.abs(self.res[1:-2] - np.ones(arr.size))
        #f = lambda x: x
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        fig.patch.set_facecolor('black')
        axs.set_facecolor('black')
        line, = axs.plot(arr, errors, color='fuchsia')
        axs.grid()
        plt.title(f"Error for $L={self.L}$, $n={self.n}$, $h={self.h:.5f}$, $s={self.s}$, "
                  f"$\\varkappa={self.kappa}$, $\\theta_r={self.theta_r}$, $I_{{\\ell}}={self.I_l}$", color='white')
        plt.xlabel(f'$x$', fontsize=12, color='white')
        plt.ylabel(f'$\\varepsilon$', fontsize=12, rotation=0, labelpad=15, color='white')
        fig.canvas.manager.set_window_title("Error")
        axs.tick_params(colors='white')

        cursor = mplcursors.cursor(line, hover=True)
        cursor.connect("add",
                   lambda sel: (
                          sel.annotation.set_text(f"$x$={sel.target[0]:.3f}\n$\\varepsilon$={sel.target[1]:.3f}"),
                          sel.annotation.set_color('white'),
                          sel.annotation.get_bbox_patch().set_facecolor("violet"),
                          sel.annotation.get_bbox_patch().set_edgecolor("purple"),
                   ))
                      

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        


if __name__ == "__main__":
    data = dict(
        L = 3,
        n = 300,
        s = 0.25,
        kappa = 0.3,
        theta_r = 0.5,
        I_l = 800
    )
    alpha = data['s'] + data['kappa']
    func = lambda x: 1 - data['I_l']/2*E2(alpha*x) - data['I_l']*data['theta_r']*E3(alpha*data['L'])*E2(alpha*(data['L']-x)) -\
                     data['s']*data['theta_r']/2*E2(alpha*(data['L']-x))*(E3(0)-E3(alpha*data['L'])) - data['s']/(2*alpha)*(2*E2(0) -\
                     E2(alpha*x) -E2(alpha*(data['L']-x)))
    
    sol = IntEq(data, func)
    #print(sol.buildMatrixA(), '\n')
    #print(sol.buildMatrixD(), '\n')
    #print(sol.buildMatrixE(), '\n')
    #print(sol.buildVectorE(), '\n')
    #print(sol.buildVectorB(), '\n')
    #print(sol.buildVectorF(), '\n')
    #print(sol.solve())
    sol.solutionGraph()
    sol.errorGraph()
