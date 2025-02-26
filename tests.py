import numpy as np
from collections import namedtuple

params = namedtuple('params', ['L', 'n', 's', 'kappa', 'theta_r', 'I_l'])

data = (
        params(L = 3.0, n = 500, s = 2.0, kappa = 2.0, theta_r = 0.0, I_l = 900.0),
        params(L = 3.0, n = 500, s = 0.2, kappa = 0.3, theta_r = 1.0, I_l = 900.0),
        params(L = 3.0, n = 500, s = 0.0, kappa = 0.3, theta_r = 1.0, I_l = 900.0), 
        params(L = 3.0, n = 500, s = 0.2, kappa = 0.0, theta_r = 1.0, I_l = 900.0),
        params(L = 3.0, n = 500, s = 2.0, kappa = 10.0, theta_r = 0.5, I_l = 900.0),
        )

colors = np.array(['fuchsia', 'cyan', 'deeppink', 'darkmagenta'])
labels = np.array(['no_zero', '$s=0$', '$\\varkappa=0$', '$\\theta_r=0$'])