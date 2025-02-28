from collections import namedtuple

import numpy as np

Params = namedtuple("Params", ["l", "n", "s", "kappa", "theta_r", "int_l"])

data = (
    Params(l=3.0, n=500, s=2.0, kappa=2.0, theta_r=0.0, int_l=900.0),
    Params(l=3.0, n=500, s=0.2, kappa=0.3, theta_r=1.0, int_l=900.0),
    Params(l=3.0, n=500, s=0.0, kappa=0.3, theta_r=1.0, int_l=900.0),
    Params(l=3.0, n=500, s=0.2, kappa=0.0, theta_r=1.0, int_l=900.0),
    Params(l=3.0, n=500, s=2.0, kappa=10.0, theta_r=0.5, int_l=900.0),
)

colors = np.array(["fuchsia", "cyan", "deeppink", "darkmagenta"])
labels = np.array(["no_zero", "$s=0$", "$\\varkappa=0$", "$\\theta_r=0$"])
