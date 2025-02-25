from intEq import IntEq
from GEI import E2, E3

from tests import data

if __name__ == "__main__":

    for test in data:
        alpha = test['s'] + test['kappa']
        func = lambda x: 1 - test['I_l']/2*E2(alpha*x) - test['I_l']*test['theta_r']*E3(alpha*test['L'])*E2(alpha*(test['L']-x)) -\
                        test['s']*test['theta_r']/alpha*E2(alpha*(test['L']-x))*(E3(0)-E3(alpha*test['L'])) - test['s']/(2*alpha)*(2*E2(0) -\
                        E2(alpha*x) -E2(alpha*(test['L']-x)))
        
        sol = IntEq(test, func)
        time = sol.solutionGraph()
        sol.errorGraph()
        print(f"Estimated time: {time}")

