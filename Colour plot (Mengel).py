# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 21:29:45 2025

@author: Timothy Heinichen
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

k1 = 0.024
k3 = 0.1
k4 = 0.016
k5 = 0.016
k6 = 0.032

k2_homeo = 0.048
k7_homeo = 0.0048

def odes(variables, t, k1, k2, k3, k4, k5, k6, k7):
    p, m, M = variables
    dpdt = k1 - k2 * M * (p / (k3 + p))
    dmdt = k4 * p**2 - k5 * m
    dMdt = k6 * m - k7 * M
    return [dpdt, dmdt, dMdt]


x0 = [0, 0, 0]
time_h = np.linspace(0, 10000, 10000)
x_h = odeint(odes, x0, time_h, args=(k1, k2_homeo, k3, k4, k5, k6, k7_homeo))
x0 = x_h[-1]  

k2_vals = np.arange(0.016, 0.048, 0.0001)
k7_vals = np.arange(0.0048, 0.016, 0.0001)  
results = np.zeros((len(k2_vals), len(k7_vals)))

time = np.linspace(0, 2880, 1440)

for i, k2 in enumerate(k2_vals):
    for j, k7 in enumerate(k7_vals):
        sol = odeint(odes, x0, time, args=(k1, k2, k3, k4, k5, k6, k7))
        p_vals = sol[:, 0]
        results[i, j] = np.max(p_vals)

plt.figure(figsize=(8,6))
plt.imshow(results.T, origin='lower',
           extent=[k2_vals[0], k2_vals[-1], k7_vals[0], k7_vals[-1]],
           aspect='auto', cmap='gist_rainbow')

clb = plt.colorbar()
clb.ax.set_title(r'Max $p53$', fontsize=12)
plt.xlabel(r'$\alpha_1$ (p53 degradation rate)', fontsize=12)
plt.ylabel(r'$\alpha_3$ (Mdm2 degradation rate)', fontsize=12)
plt.title(r'p53 response to variation of $\alpha_1$ and $\alpha_3$', fontsize=14)
plt.show()