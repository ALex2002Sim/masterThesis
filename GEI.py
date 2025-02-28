import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expn, factorial  # type: ignore

plt.style.use("seaborn-v0_8-darkgrid")


def exponential_integral(n: int, x_val: float) -> float:
    """
    Computes the Generalized Exponential Integral (GEI).

    Parameters:
    -----------
    n : int
        Order (must be ≥ 0).
    x_val : float
        Argument (must be ≥ 0).

    Returns:
    -----------
        float: The computed GEI value.
    """

    def psi(n_val: int) -> float:
        if n_val == 1:
            return -np.euler_gamma
        return -np.euler_gamma + sum(1 / i for i in range(1, n_val))

    def series_mem(x_v: float, m_v: int, n_v: int) -> float:
        return float(
            np.power(-x_v, m_v) * np.power((m_v - n_v + 1) * factorial(m_v), -1)
        )

    if x_val < 0 or n < 0:
        raise ValueError("x and n must be non-negative")

    if n == 0:
        return float(np.exp(-x_val) / x_val)

    if x_val == 0:
        return np.inf if n == 1 else 1 / (n - 1)

    if x_val <= 1.0:
        res = np.power(-x_val, n - 1) / factorial(n - 1) * (-np.log(x_val) + psi(n))
        m, eps, ratio = 0, 1.0e-18, 10.0

        while ratio > eps:
            if m != n - 1:
                res -= series_mem(x_val, m, n)
                ratio = abs(series_mem(x_val, m, n) / res)
            m += 1

        return float(res)

    m, b, c = 1, x_val + n, 1.0e30
    d, f, eps, ratio = 1.0 / b, 1.0 / b, 1.0e-18, 10.0

    while ratio > eps:
        a = -m * (n - 1 + m)
        b += 2.0
        d = 1.0 / (a * d + b)
        c = b + a / c
        fact = c * d
        f *= fact
        m += 1
        ratio = abs(fact - 1.0)

    return float(np.exp(-x_val) * f)


def e1(x: float) -> float:
    return exponential_integral(1, x)


def e2(x: float) -> float:
    return exponential_integral(2, x)


def e3(x: float) -> float:
    return exponential_integral(3, x)


def e4(x: float) -> float:
    return exponential_integral(4, x)


if __name__ == "__main__":
    path = os.path.join("graphs", "GEI.png")
    os.makedirs("graphs", exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    arr = np.linspace(0.00001, 4, 1000)
    ind = np.arange(1, 5).astype(int)
    col = ["fuchsia", "cyan", "deeppink", "blueviolet"]
    style = ["-", "-.", ":", "--"]

    for i in ind:
        e_val = np.array([exponential_integral(i, x) for x in arr])
        axs[0].plot(
            arr,
            e_val,
            color=col[i - 1],
            label=f"$E_{i}(x)$",
            ls=style[i - 1],
            linewidth=2,
        )

    for i in ind:
        expn_arr = expn(i, arr)
        axs[1].plot(
            arr,
            expn_arr,
            color=col[i - 1],
            label=f"$E_{i}(x)$",
            ls=style[i - 1],
            linewidth=2,
        )

    manager = plt.get_current_fig_manager()
    if manager is not None:
        manager.set_window_title("Generalized Exponential Integral")

    axs[0].set_title("My $E_n$")
    axs[1].set_title("$E_n$ from $scipy$")

    for ax in axs:
        ax.set_ylim(0, 1.5)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_ylabel("$E_n$", fontsize=12, rotation=0, labelpad=15)

    fig.suptitle("Tests for $E_n(x)$")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
