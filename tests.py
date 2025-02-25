import numpy as np

data = (
        dict(L = 3, n = 1000, s = 0.2, kappa = 0.3, theta_r = 0.5, I_l = 900),
        dict(L = 3, n = 1000, s = 0.0, kappa = 0.3, theta_r = 0.5, I_l = 900), 
        dict(L = 3, n = 1000, s = 0.2, kappa = 0.0, theta_r = 0.5, I_l = 900),
        dict(L = 3, n = 1000, s = 0.2, kappa = 0.3, theta_r = 0.0, I_l = 900)
        )

colors = np.array(['fuchsia', 'cyan', 'deeppink', 'darkmagenta'])
labels = np.array(['no_zero', '$s=0$', '$\\varkappa=0$', '$\\theta_r=0$'])