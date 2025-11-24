# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:20:13 2024

@author: Timothy Heinichen
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks

# Define constants
k1 = 0.024
k2_d = 0.016
k3 = 0.1
k4 = 0.016
k5 = 0.016
k6 = 0.032
k7_d = 0.016
k9 = 0.016
k10 = 10

# Initial conditions for p, m, M, and D
initial_conditions = [0, 0, 0, 0]

# Define the system of differential equations
def odes(variables, t, k8, k2, k7):
    p, m, M, D = variables
    
    dpdt = k1 - k2 * M * (p / (k3 + p)) * (k10 / (D + k10))
    dmdt = k4 * p**2 - k5 * m
    dMdt = k6 * m - k7 * M
    dDdt = k8 * p**2 - k9 * D
    
    return [dpdt, dmdt, dMdt, dDdt]

time_h = np.linspace(0, 10000, 10000)  
time = np.linspace(0,2880,1440)
k8_values = np.arange(0,0.16,0.001)

amplitudes_p = []
periods_p = []
amplitudes_D = []
periods_D = []

for k8 in k8_values:
   
    k2_h = 0.048
    k7_h= 0.0048
    solution_steady = odeint(odes, initial_conditions, time_h, args=(k8, k2_h, k7_h))
    steady_state_values = solution_steady[-1, :]
    new_initial_conditions = steady_state_values
    
    k2 = k2_d
    k7 = k7_d

    solution = odeint(odes, new_initial_conditions, time, args=(k8, k2, k7))

    p = solution[:, 0]
    D = solution[:, 3]
    
    peaks_p, _ = find_peaks(p)
    troughs_p, _ = find_peaks(-p)
    peaks_D, _ = find_peaks(D)
    troughs_D, _ = find_peaks(-D)
    
    if len(peaks_p) > 0 and len(troughs_p) > 0:
        peak_amplitudes_p = p[peaks_p]
        trough_amplitudes_p = p[troughs_p]
        min_length_p = min(len(peak_amplitudes_p), len(trough_amplitudes_p))
        amplitude_p = np.mean(peak_amplitudes_p[:min_length_p] - trough_amplitudes_p[:min_length_p])
        amplitudes_p.append(amplitude_p)

        peak_periods_p = np.diff(time[peaks_p])
        period_p = np.mean(peak_periods_p) if len(peak_periods_p) > 0 else np.nan
        periods_p.append(period_p)
    else:
        amplitudes_p.append(np.nan)
        periods_p.append(np.nan)
    
    if len(peaks_D) > 0 and len(troughs_D) > 0:
        peak_amplitudes_D = D[peaks_D]
        trough_amplitudes_D = D[troughs_D]
        min_length_D = min(len(peak_amplitudes_D), len(trough_amplitudes_D))
        amplitude_D = np.mean(peak_amplitudes_D[:min_length_D] - trough_amplitudes_D[:min_length_D])
        amplitudes_D.append(amplitude_D)
        
        # Calculate periods for D
        peak_periods_D = np.diff(time[peaks_D])
        period_D = np.mean(peak_periods_D) if len(peak_periods_D) > 0 else np.nan
        periods_D.append(period_D)
    else:
        amplitudes_D.append(np.nan)
        periods_D.append(np.nan)

# Plotting the amplitude and period variation for p and D
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot for p
ax1.set_xlabel(r'$\alpha_4$')
ax1.set_ylabel('Amplitude', color='tab:blue')
ax1.plot(k8_values, amplitudes_p, '-', color='tab:blue', label='Amplitude of p')  # Use '-' for solid line
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax1_2 = ax1.twinx()
ax1_2.set_ylabel('Period (min)', color='tab:red')
ax1_2.plot(k8_values, periods_p, '-', color='tab:red', label='Period of p')  # Use '-' for solid line
ax1_2.tick_params(axis='y', labelcolor='tab:red')

ax1.set_title(r'Variation of Amplitude and Period for p53 in function of $\alpha_4$')

# Plot for D
ax2.set_xlabel(r'$\alpha_4$')
ax2.set_ylabel('Amplitude', color='tab:blue')
ax2.plot(k8_values, amplitudes_D, '-', color='tab:blue', label='Amplitude of D')  # Use '-' for solid line
ax2.tick_params(axis='y', labelcolor='tab:blue')

ax2_2 = ax2.twinx()
ax2_2.set_ylabel('Period (min)', color='tab:red')
ax2_2.plot(k8_values, periods_D, '-', color='tab:red', label='Period of D')  # Use '-' for solid line
ax2_2.tick_params(axis='y', labelcolor='tab:red')

ax2.set_title(r'Variation of Amplitude and Period for DINO in function of $\alpha_4$')

fig.tight_layout()
plt.show()
