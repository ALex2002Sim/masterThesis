from intEq import IntEq
from GEI import E2, E3

if __name__ == "__main__":
    data = dict(
        L = 1,
        n = 500,
        s = 0.1,
        kappa = 0.1,
        theta_r = 0.0,
        I_l = 900
    )
    
    alpha = data['s'] + data['kappa']
    func = lambda x: 1 - data['I_l']/2*E2(alpha*x) - data['I_l']*data['theta_r']*E3(alpha*data['L'])*E2(alpha*(data['L']-x)) -\
                     data['s']*data['theta_r']/alpha*E2(alpha*(data['L']-x))*(E3(0)-E3(alpha*data['L'])) - data['s']/(2*alpha)*(2*E2(0) -\
                     E2(alpha*x) -E2(alpha*(data['L']-x)))
    
    sol = IntEq(data, func)
    sol.solutionGraph()
    sol.errorGraph()

