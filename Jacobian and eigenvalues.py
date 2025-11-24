# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 09:14:36 2025

@author: Timothy Heinichen
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve

k2_homeo = 0.048
k7_homeo = 0.0048

k1  = 0.024
k2  = 0.016      
k3  = 0.1
k4  = 0.016
k5  = 0.016
k6  = 0.032
k7  = 0.016
k8  = 0.036
k9  = 0.016
k10 = 10.0


def odes(vars, t, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10):
    p, m, M, D = vars
    dpdt = k1 - k2 * M * (p / (k3 + p)) * (k10 / (D + k10))
    dmdt = k4 * p**2 - k5 * m
    dMdt = k6 * m - k7 * M
    dDdt = k8 * p**2 - k9 * D
    return [dpdt, dmdt, dMdt, dDdt]

def steady_state(k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, guess=[1, 1, 1, 1]):
    func = lambda vars: odes(vars, 0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10)
    p_star, m_star, M_star, D_star = fsolve(func, guess)
    return p_star, m_star, M_star, D_star


def jacobian(p, m, M, D, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10):
    # ---- dp/dt partials ----
    df1_dp = -k2 * M * (k3 / (k3 + p)**2) * (k10 / (D + k10))          
    df1_dm = 0.0                                                       
    df1_dM = -k2 * (p / (k3 + p)) * (k10 / (D + k10))                  
    df1_dD =  k2 * M * (p / (k3 + p)) * (k10 / (D + k10)**2)           

    # ---- dm/dt partials ----
    df2_dp =  2.0 * k4 * p                                             
    df2_dm = -k5                                                       
    df2_dM =  0.0                                                      
    df2_dD =  0.0                                                      

    # ---- dM/dt partials ----
    df3_dp =  0.0                                                      
    df3_dm =  k6                                                       
    df3_dM = -k7                                                       
    df3_dD =  0.0                                                      

    # ---- dD/dt partials ----
    df4_dp =  2.0 * k8 * p                                             
    df4_dm =  0.0                                                      
    df4_dM =  0.0                                                      
    df4_dD = -k9                                                       

    J = np.array([
        [df1_dp, df1_dm, df1_dM, df1_dD],
        [df2_dp, df2_dm, df2_dM, df2_dD],
        [df3_dp, df3_dm, df3_dM, df3_dD],
        [df4_dp, df4_dm, df4_dM, df4_dD],
    ], dtype=float)

    return J

def interpret_eigs(eigvals):
    if np.any(np.abs(eigvals.imag) > 1e-9):
        if np.all(eigvals.real < 0):
            return "Damped oscillations (stable spiral/focus)"
        elif np.any(eigvals.real > 0):
            return "Growing oscillations (unstable spiral/focus)"
        else:
            return "Neutral oscillations (center-like)"
    else:
        if np.all(eigvals.real < 0):
            return "Stable node (no oscillations)"
        elif np.any(eigvals.real > 0):
            return "Saddle/unstable node"
        else:
            return "Marginal case"

def analyze_regime(name, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, guess=[1,1,1,1]):
    print(f"\n=== {name} ===")
    p_star, m_star, M_star, D_star = steady_state(k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, guess=guess)
    print(f"Fixed point: p* = {p_star:.6f}, m* = {m_star:.6f}, M* = {M_star:.6f}, D* = {D_star:.6f}")

    J = jacobian(p_star, m_star, M_star, D_star, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10)
    print("Jacobian matrix:\n", J)

    eigvals = np.linalg.eigvals(J)
    for i, val in enumerate(eigvals, 1):
        print(f"λ{i} = {val.real:.6f} {'+' if val.imag>=0 else '-'} {abs(val.imag):.6f}i")

    print("→", interpret_eigs(eigvals))

analyze_regime("Homeostasis",
               k1, k2_homeo, k3, k4, k5, k6, k7_homeo,
               k8, k9, k10,
               guess=[1,1,1,1])

analyze_regime("DNA damage",
               k1, k2, k3, k4, k5, k6, k7,
               k8, k9, k10,
               guess=[1,1,1,1])

