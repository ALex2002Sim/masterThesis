import time

from int_eq import IntEq
from gei import e2, e3
from tests import data
from graph import draw_graph

if __name__ == "__main__":

    for i, test in enumerate(data):
        l, n, s, kappa, theta_r, int_l = test
        alpha = s + kappa
        func = (
            lambda x: 1
            - int_l / 2 * e2(alpha * x)
            - int_l * theta_r * e3(alpha * l) * e2(alpha * (l - x))
            - s * theta_r / alpha * e2(alpha * (l - x)) * (e3(0) - e3(alpha * l))
            - s / (2 * alpha) * (2 * e2(0) - e2(alpha * x) - e2(alpha * (l - x)))
        )

        sol = IntEq(test, func)

        start = time.perf_counter()
        res = sol.solve()
        end = time.perf_counter()

        draw_graph(res, test, "Solution")
        draw_graph(res, test, "Error")

        print(f"Test {i+1} done...\nEstimated time: {end-start:.4f} s\n")
