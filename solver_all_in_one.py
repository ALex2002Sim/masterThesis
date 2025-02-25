from intEq import IntEq
from GEI import E2, E3
from matplotlib import pyplot as plt
import numpy as np

from tests import data, colors, labels

plt.style.use('seaborn-v0_8-darkgrid')

if __name__=="__main__":
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    for i, test in enumerate(data):
        alpha = test['s'] + test['kappa']
        func = lambda x: 1 - test['I_l']/2*E2(alpha*x) - test['I_l']*test['theta_r']*E3(alpha*test['L'])*E2(alpha*(test['L']-x)) -\
                        test['s']*test['theta_r']/alpha*E2(alpha*(test['L']-x))*(E3(0)-E3(alpha*test['L'])) - test['s']/(2*alpha)*(2*E2(0) -\
                        E2(alpha*x) -E2(alpha*(test['L']-x)))
        
        sol = IntEq(test, func)
        res = sol.solve()

        axs.plot(np.linspace(0, test['L'], res.size)[1:-2], res[1:-2], color=colors[i], label=labels[i], linewidth=2)
    
    fig.canvas.manager.set_window_title("All solutions")
    plt.xlabel(f'$x$', fontsize=12)
    plt.ylabel(f'$\\mathcal{{S}}(x)$', fontsize=12, rotation=0, labelpad=20)
    axs.grid(True, linestyle='--', alpha=0.6)
    axs.legend()

    plt.savefig("graphs/all_solutions.png", dpi=300, bbox_inches='tight')
    plt.show()