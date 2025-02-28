from typing import Callable, Tuple
import numpy as np
from scipy.integrate import quad
from scipy.linalg import toeplitz

from gei import e3
from tests import Params


class IntEq:
    """
    Class for solving the integral equation of radiative transfer.

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

    def __init__(self, d: Params, func: Callable[[float], float]) -> None:
        self.l = d.l
        self.n = d.n
        self.s = d.s
        self.kappa = d.kappa
        self.alpha = self.kappa + self.s
        self.theta_r = d.theta_r
        self.int_l = d.int_l
        self.h = self.l / self.n

        self.mtr_a = np.zeros((self.n, self.n))
        self.mtr_e = np.zeros((self.n, self.n))
        self.mtr_d = np.zeros((self.n, self.n))
        self.e = np.zeros(self.n)
        self.b = np.zeros(self.n)
        self.func = func
        self.f = np.zeros(self.n)

        self.res = np.zeros(self.n)

    def x(self, j: int) -> float:
        return float(
            0 if j == 0 else self.l if j == self.n else j * self.h - self.h / 2
        )

    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.fill_diagonal(self.mtr_a, self.h)
        self.mtr_a[0, 0] = self.h / 2
        self.mtr_a[-1, -1] = self.mtr_a[0, 0]

        vals = np.array([self.x(i) for i in np.arange(self.n + 1)])
        my_en_vals = np.array([e3(self.alpha * (self.l - el)) for el in vals])
        diff_vals = np.diff(my_en_vals)
        self.mtr_d = np.outer(diff_vals, diff_vals)

        c = np.power(self.alpha, -2)
        a = self.alpha * self.h
        b = self.h * np.power(self.alpha, -1)
        ind = np.arange(1, self.n)
        arr = np.zeros(self.n)
        for i in ind[1:]:
            arr[i - 1] = e3(a * (i)) + e3(a * (i - 2)) - 2 * e3(a * (i - 1))
        arr *= c
        arr[0] = 2 * b + 2 * c * e3(a) - c
        self.mtr_e = toeplitz(arr)
        arr = np.zeros(self.n)
        for i in ind[1:]:
            arr[i - 1] = (
                -e3(a * (i - 1))
                + e3(a * (i - 2))
                + e3(a * (i - 0.5))
                - e3(a * (i - 1.5))
            )
        self.mtr_e[:, 0] = c * arr
        arr = np.zeros(self.n)
        for i in ind[:-1]:
            arr[i - 1] = (
                -e3(self.alpha * (self.l + self.h * (0.5 - i)))
                + e3(self.alpha * (self.l + self.h * (1.5 - i)))
                + e3(self.alpha * (self.l - i * self.h))
                - e3(self.alpha * (self.l - self.h * (i - 1)))
            )
        self.mtr_e[:, -1] = c * arr
        self.mtr_e[0, 0] = b + 2 * c * e3(a / 2) - c
        self.mtr_e[-1, -1] = self.mtr_e[0, 0]

        return self.mtr_a, np.power(self.alpha, -2) * self.mtr_d, self.mtr_e

    def build_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for i in range(1, self.n + 1):
            self.e[i - 1] = e3(self.alpha * self.x(i)) - e3(self.alpha * self.x(i - 1))

        for i in range(1, self.n + 1):
            self.b[i - 1] = e3(self.alpha * (self.l - self.x(i))) - e3(
                self.alpha * (self.l - self.x(i - 1))
            )

        for i in range(1, self.n + 1):
            self.f[i - 1] = quad(self.func, self.x(i - 1), self.x(i))[0]

        return (
            -np.power(self.alpha, -1) * self.e,
            np.power(self.alpha, -1) * self.b,
            self.f,
        )

    def solve(self) -> np.ndarray:
        self.mtr_a, self.mtr_d, self.mtr_e = self.build_matrices()
        self.e, self.b, self.f = self.build_vectors()

        mtr_b = (
            self.mtr_a - self.s * self.theta_r * self.mtr_d - self.s / 2 * self.mtr_e
        )
        k = (
            self.int_l / 2 * self.e
            + self.int_l * self.theta_r * e3(self.alpha * self.l) * self.b
            + self.f
        )

        self.res = np.linalg.solve(mtr_b, k)

        return self.res
