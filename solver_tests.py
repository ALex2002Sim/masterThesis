from intEq import IntEq
from GEI import E2, E3
import time

from tests import data
from graph import draw_graph

if __name__ == "__main__":

    for i, test in enumerate(data):
        L, n, s, kappa, theta_r, I_l = test
        alpha = s + kappa
        func = lambda x: 1 - I_l/2*E2(alpha*x) - I_l*theta_r*E3(alpha*L)*E2(alpha*(L-x)) -\
                        s*theta_r/alpha*E2(alpha*(L-x))*(E3(0)-E3(alpha*L)) - s/(2*alpha)*(2*E2(0) - E2(alpha*x) - E2(alpha*(L-x)))
        
        sol = IntEq(test, func)

        start = time.perf_counter()
        res = sol.solve()
        end = time.perf_counter()

        draw_graph(res, test, "Solution")
        draw_graph(res, test, "Error")

        print(f"Test {i+1} done...\nEstimated time: {end-start:.4f} s\n")