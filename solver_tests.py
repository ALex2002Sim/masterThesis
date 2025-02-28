import time
from typing import Callable

from gei import e2, e3
from graph import draw_graph
from int_eq import IntEq
from tests import data


def create_func(
    il: float, alpha: float, th_r: float, leng: float, s_: float
) -> Callable[[float], float]:
    return (
        lambda x: 1
        - il / 2 * e2(alpha * x)
        - il * th_r * e3(alpha * leng) * e2(alpha * (leng - x))
        - s_ * th_r / alpha * e2(alpha * (leng - x)) * (e3(0) - e3(alpha * leng))
        - s_ / (2 * alpha) * (2 * e2(0) - e2(alpha * x) - e2(alpha * (leng - x)))
    )


if __name__ == "__main__":

    for i, test in enumerate(data):
        l, n, s, kappa, theta_r, int_l = test

        func = create_func(int_l, s + kappa, theta_r, l, s)

        sol = IntEq(test, func)

        start = time.perf_counter()
        res = sol.solve()
        end = time.perf_counter()

        draw_graph(res, test, "Solution")
        draw_graph(res, test, "Error")

        print(f"Test {i+1} done...\nEstimated time: {end-start:.4f} s\n")
