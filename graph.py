import os
import numpy as np
from matplotlib import pyplot as plt
from tests import Params

plt.style.use("seaborn-v0_8-darkgrid")


def draw_graph(res: np.ndarray, data: Params, graph_type: str) -> None:
    y_label = {"Error": "$\\varepsilon$", "Solution": "$\\mathcal{{S}}(x)$"}
    folder = "graphs"
    l, n, s, kappa, theta_r, int_l = data
    h = l / n

    res = res[1:-2].copy()

    _, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel(y_label[graph_type], fontsize=12, rotation=0, labelpad=15)
    plt.title(
        f"{graph_type} for $L={l}$, $n={n}$, $h={h:.5f}$, $s={s}$, "
        f"$\\varkappa={kappa}$, $\\theta_r={theta_r}$, $I_{{\\ell}}={int_l}$"
    )
    manager = plt.get_current_fig_manager()
    if manager is not None:
        manager.set_window_title("Generalized Exponential Integral")
    file_name = f"{graph_type.lower()}s_/{graph_type[:3]}_h={h}_s={s}_k={kappa}_thR={theta_r}_I={int_l}.png"
    path = os.path.join(folder, file_name)

    if graph_type == "Error":
        res = np.abs(res - np.ones(res.size)).copy()

    axs.plot(np.linspace(0, l, res.size), res, color="deeppink", linewidth=2)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    # plt.show()
