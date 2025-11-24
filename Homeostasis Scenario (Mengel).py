# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 21:22:51 2025

@author: Timothy Heinichen
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

k2_homeo = 0.048
k7_homeo = 0.0048

k1 = 0.024
k2 = 0.016
k3 = 0.1
k4 = 0.016
k5 = 0.016
k6 = 0.032
k7 = 0.016

def odes(variables,time, k1, k2, k3, k4, k5, k6, k7):
    p, m, M = variables
    
    dpdt = k1 - k2 * M * (p / (k3 + p))
    dmdt = k4 * p**2 - k5 * m
    dMdt = k6 * m - k7 * M
    
    return [dpdt, dmdt, dMdt]

# Initial conditions
x0 = [0, 0, 0]
time = np.linspace(0,2880,1440)

x= odeint(odes, x0, time, args=(k1, k2_homeo, k3, k4, k5, k6, k7_homeo))
p = x[:,0]
m = x[:,1]
M = x[: ,2]


plt.figure(figsize=(12, 6))
plt.plot(time, x[:, 0], '-', label='p53')
plt.plot(time, x[:, 1], '-', label='Mdm2 mRNA')
plt.plot(time, x[:, 2], '-', label='Mdm2')

plt.legend()
plt.xlabel('Time (min)')
plt.ylabel('Levels (a.u)')
plt.title('p53 dynamics in homeostasis')
plt.grid(True)
plt.show()