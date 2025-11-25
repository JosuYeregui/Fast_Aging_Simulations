import numpy as np

def cast_params_for_SEI(param):
    pass

def get_ldif(L0, t, input_vars):

    L = integration(L0, t, input_vars)
    return L - L0

def integration(L0, t, input_vars):
    """
    Piecewise integration for dL/dt = A*D*k/(D + k*L)
    treating nearly-constant k segments analytically.
    """
    k_exp_t = input_vars["k_exp_t"]
    D = input_vars["D"]
    A = input_vars["A"]

    dt = 1.

    # Normal small-step exact update with local k
    k_mid = 0.5 * (k_exp_t(t) + k_exp_t(t + dt))
    L= (-D + np.sqrt(D**2 + 2*k_mid*(D*L0 + 0.5*k_mid*L0**2 + A*D*k_mid*dt))) / k_mid
    return L