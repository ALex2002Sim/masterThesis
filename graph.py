import numpy as np
from matplotlib import pyplot as plt
import os
from tests import params

plt.style.use('seaborn-v0_8-darkgrid')

def draw_graph(res:np.ndarray, data:params, graph_type:str)->None:
    y_label = {"Error": "$\\varepsilon$", "Solution": "$\\mathcal{{S}}(x)$"}
    folder = "graphs"
    L, n, s, kappa, theta_r, I_l = data
    h = L/n

    res = res[1:-2].copy()

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel(y_label[graph_type], fontsize=12, rotation=0, labelpad=15)
    plt.title(f"{graph_type} for $L={L}$, $n={n}$, $h={h:.5f}$, $s={s}$, "
                  f"$\\varkappa={kappa}$, $\\theta_r={theta_r}$, $I_{{\\ell}}={I_l}$")
    fig.canvas.manager.set_window_title(f"{graph_type}")
    path = os.path.join(folder, f'{graph_type.lower()}s_/{graph_type[:3]}_h={h}_s={s}_k={kappa}_thR={theta_r}_I={I_l}.png')

    if graph_type == "Error":
        res = np.abs(res - np.ones(res.size)).copy()

    axs.plot(np.linspace(0, L, res.size), res, color='deeppink', linewidth=2)

    plt.savefig(path, dpi=300, bbox_inches='tight')
    #plt.show()