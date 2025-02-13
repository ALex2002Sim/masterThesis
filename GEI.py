import numpy as np
from scipy.special import factorial, expn
from matplotlib import pyplot as plt
import os


def En(n:np.int64, x:np.float64)->np.float64:

    """
    generalized exponential integral (GEI) for integer n >= 0 and real x >= 0

    Parameters
    ----------
    n: int
        order of GEI
    x: float
        argument of GEI
    
    Returns
    ----------
    scalar
        value of GEI
    """

    def psi(n:np.int64)->np.float64:
        return -np.euler_gamma if n==1 else -np.euler_gamma + sum([1/i for i in range(1, n)])

    def seriesMem(x:np.float64, m:np.int64, n:np.int64)->np.float64:
        return np.power(-x, m)*np.power((m - n + 1)*factorial(m), -1)

    try:
        if x < 0:
            res = np.nan
            raise ValueError("Negative argument")
        if n < 0:
            res = np.nan
            raise ValueError("Negative order")
        if n == 0:
            res = np.exp(-x)*np.power(x, -1)

        else:
            eps, ratio = 1.0e-18, 10.0
            if x == 0:
                res = np.inf if n==1 else 1/(n-1)
            else:
                if x <= 1.0:
                    res = np.power(-x, n-1)*np.power(factorial(n-1), -1)*(-np.log(x) + psi(n))
                    m = 0
                    while ratio > eps:
                        if m != n-1:
                            res -= seriesMem(x, m, n)
                            ratio = np.abs(seriesMem(x, m, n)*np.power(res, -1))
                        m += 1

                else:
                    m, b, c = 1.0, x+n, 1.0*np.power(1.0e-30, -1)
                    d = 1.0*np.power(b, -1)
                    f = d

                    while ratio > eps:
                        a = -m*(n-1+m)
                        b += 2.0
                        d = 1.0*np.power((a*d + b), -1)
                        c = b + a*np.power(c, -1)
                        fact = c*d
                        f *= fact
                        m += 1
                        ratio = np.abs(fact - 1.0)
                    
                    res = np.exp(-x)*f

        return res
    
    except ValueError as e:
        print(f"Error: {str(e)}")

E1 = lambda x: En(1, x)
E2 = lambda x: En(2, x)
E3 = lambda x: En(3, x)
E4 = lambda x: En(4, x)


if __name__ == "__main__":
    path = os.path.join('graphs', 'GEI.png')
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    arr = np.linspace(0.00001, 4, 1000)
    ind = np.int64(np.linspace(1, 4, 4))
    col = np.array(['fuchsia', 'cyan', 'deeppink', 'blueviolet'])
    style = np.array(['-', '-.', ':', '--'])

    for i in ind:
        Eval = np.zeros(arr.size)
        for j, x in enumerate(arr):
            Eval[j] = En(i, x)
        axs[0].plot(arr, Eval, color=col[i-1], label=f'$E_{i}(x)$', ls=style[i-1])

    for i in ind:
        axs[1].plot(arr, expn(i, arr), color=col[i-1], label=f'$E_{i}(x)$', ls=style[i-1])

    for ax in axs:
        ax.set_ylim(0, 1.5)
        ax.grid()
        ax.legend()
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$E_n$', fontsize=12, rotation=0, labelpad=15)

    axs[0].set_title('My GEI')
    axs[1].set_title('GEI from scipy')

    fig.suptitle('Tests for $E_n(x)$')
    fig.canvas.manager.set_window_title("GEI")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

    for i in ind:
        Eval = np.zeros(arr.size)
        for j, x, in enumerate(arr):
            Eval[j] = En(i, x)
        print(f'Error of my E_{i}: {np.linalg.norm(Eval-expn(i, arr), ord=2)}')