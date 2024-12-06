import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
from typing import Callable
import mplcursors
import os

from GEI import E2, E3, E4

class IntEq:

    def __init__(self, L:np.int64, n:np.int64, s:np.float64, kappa:np.float64, theta_r:np.float64, I_l:np.float64):
        self.L = L
        self.n = n
        self.s = s
        self.kappa = kappa
        self.alpha = self.kappa + self.s
        self.theta_r = theta_r
        self.I_l = I_l
        self.h = self.L/self.n

        self.A = np.zeros((self.n, self.n))
        self.E = np.zeros((self.n, self.n))
        self.D = np.zeros((self.n, self.n))
        self.e = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.fr = np.zeros(self.n)

        self.res = np.zeros(self.n)

    func = lambda self, x: 1 - self.I_l/2*E2(self.alpha*x) - self.I_l*self.theta_r*E3(self.alpha*self.L)*E2(self.alpha*(self.L-x)) -\
                         self.s*self.theta_r/2*E2(self.alpha*(self.L-x))*(E3(0)-E3(self.alpha*self.L)) - self.s/(2*self.alpha)*(2*E2(0) - E2(self.alpha*x) - E2(self.alpha*(self.L-x)))

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
            self.fr[i-1] = quad(self.func, self.x(i-1), self.x(i))[0]

        return self.fr
    
    def solve(self)->np.ndarray:
        self.D = self.buildMatrixD()
        self.A = self.buildMatrixA()
        self.E = self.buildMatrixE()
        self.b = self.buildVectorB()
        self.e = self.buildVectorE()
        self.fr = self.buildVectorF()

        B = self.A - self.s*self.theta_r*self.D - self.s/2*self.E
        k = self.I_l/2*self.e + self.I_l*self.theta_r*E3(self.alpha*self.L)*self.b + self.fr

        self.res = np.linalg.solve(B, k)

        return self.res
    
    def solutionGraph(self, filename:str='solution')->None:
        path = os.path.join('graphs', f'{filename}.png')
        if not os.path.exists('graphs'):
            os.makedirs('graphs')

        self.res = self.solve()
        arr = np.linspace(0, self.L, self.res.size)
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        line, = axs.plot(arr[1:-2], self.res[1:-2], color='fuchsia')
        axs.grid()
        plt.title(f"Solution for $L={self.L}$, $n={self.n}$, $s={self.s}$, "
                  f"$\\varkappa={self.kappa}$, $\\theta_r={self.theta_r}$, $I_{{\\ell}}={self.I_l}$")
        plt.xlabel(f'$x$', fontsize=12)
        plt.ylabel(f'$\\mathcal{{S}}(x)$', fontsize=12, rotation=0, labelpad=20)
        fig.canvas.manager.set_window_title("Solution")

        cursor = mplcursors.cursor(line, hover=True)
        cursor.connect("add",
                        lambda sel: (
                               sel.annotation.set_text(f"$x$={sel.target[0]:.3f}\n$\\mathcal{{S}}$={sel.target[1]:.3f}"),
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
        line, = axs.plot(arr, errors, color='fuchsia')
        axs.grid()
        plt.title(f"Error for $L={self.L}$, $n={self.n}$, $s={self.s}$, "
                  f"$\\varkappa={self.kappa}$, $\\theta_r={self.theta_r}$, $I_{{\\ell}}={self.I_l}$")
        plt.xlabel(f'$x$', fontsize=12)
        plt.ylabel(f'$\\varepsilon$', fontsize=12, rotation=0, labelpad=15)
        fig.canvas.manager.set_window_title("Error")

        cursor = mplcursors.cursor(line, hover=True)
        cursor.connect("add",
                   lambda sel: (
                          sel.annotation.set_text(f"$x$={sel.target[0]:.3f}\n$\\varepsilon$={sel.target[1]:.3f}"),
                          sel.annotation.get_bbox_patch().set_facecolor("violet"),
                          sel.annotation.get_bbox_patch().set_edgecolor("purple"),
                   ))
                      

        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
        


if __name__ == "__main__":
    sol = IntEq(1, 5, 0.25, 0.3, 0.5, 800)
    #print(sol.buildMatrixA(), '\n')
    #print(sol.buildMatrixD(), '\n')
    #print(sol.buildMatrixE(), '\n')
    #print(sol.buildVectorE(), '\n')
    #print(sol.buildVectorB(), '\n')
    #print(sol.buildVectorF(), '\n')
    #print(sol.solve())
    sol.solutionGraph()
    #sol.errorGraph()
