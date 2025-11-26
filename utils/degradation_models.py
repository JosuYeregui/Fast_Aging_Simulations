import numpy as np
import pybamm

def get_dif(L0, t, cycler, input_vars):

    L = tfunc_integration_LSEI(L0, t, cycler, input_vars)
    # L = piecewise_integration_LSEI(L0, cycler, input_vars)
    return L - L0

def cast_params_for_LAM(param, input_vars):
    return input_vars

def cast_params_for_SEI(param, input_vars):

    k_sei = float(param.get("SEI kinetic rate constant [m.s-1]", 1e-12))
    D_ec = float(param.get("EC diffusivity [m2.s-1]", 2e-18))  # or "Molar mass of SEI"
    c_ec = float(param.get("EC initial concentration in electrolyte [mol.m-3]", 4541))
    alpha = float(param.get("SEI charge transfer coefficient", 0.5))
    U_sei = float(param.get("SEI open-circuit potential [V]", 0.4))
    F = float(param.get("Faraday constant [C.mol-1]", 96485.33212))
    R = float(param.get("Ideal gas constant [J.mol-1.K-1]", 8.314462618))
    T = float(param.get("Ambient temperature [K]", 298.15))
    V_sei = float(param.get("SEI partial molar volume", 9.585e-5))
    x_0, x_100, _, _ = pybamm.lithium_ion.get_min_max_stoichiometries(param)
    OCP_n = param["Negative electrode OCP [V]"]

    def k_exp(soc):

        #sol_n = (soc - x_0) / (x_100 - x_0)
        sol_n = soc * (x_100 - x_0) + x_0

        #  = np.tile(SOL_n_cycle, n_cycles * 2)
        eta = OCP_n(sol_n) - U_sei

        exp_vals = np.exp(-alpha * F * eta / (R * T))
        k_exp_val = exp_vals * k_sei
        return k_exp_val

    A = c_ec * V_sei / 2

    input_vars["k_exp"] = k_exp
    input_vars["A"] = A
    input_vars["D"] = D_ec

    return input_vars

def tfunc_integration_LSEI(L0, t, cycler, input_vars):
    """
    Piecewise integration for dL/dt = A*D*k/(D + k*L)
    treating nearly-constant k segments analytically.
    """
    k_exp_t = input_vars["k_exp"]
    D = input_vars["D"]
    A = input_vars["A"]

    dt = 1.

    _, soc = cycler(t/3600.)
    _, soc_dt = cycler((t + dt)/3600.)

    # Normal small-step exact update with local k
    k_mid = 0.5 * (k_exp_t(soc) + k_exp_t(soc_dt))
    L= (-D + np.sqrt(D**2 + 2*k_mid*(D*L0 + 0.5*k_mid*L0**2 + A*D*k_mid*dt))) / k_mid
    return L

def piecewise_integration_LSEI(L0, cycler, input_vars, slope_tol=1e-3, min_len=5):
    """
    Piecewise integration for dL/dt = A*D*k/(D + k*L)
    treating nearly-constant k segments analytically.
    """
    k_exp_fn = input_vars["k_exp"]
    D = input_vars["D"]
    A = input_vars["A"]

    t = np.arange(0, cycler.total_time*3600., 10)

    k_exp_t = []
    for t_ in t:
        _, soc_ = cycler(t_/3600.)
        k_exp_t.append(k_exp_fn(soc_))

    dt = np.diff(t)

    # Compute relative slope of k_exp_t to detect flat regions ("plateaus")
    dk = np.gradient(k_exp_t, t,)
    flat = np.abs(dk / np.maximum(np.abs(k_exp_t), 1e-30)) < slope_tol

    # Identify contiguous flat regions
    segments = []
    i = 0
    N = len(t)
    while i < N - 1:
        if flat[i]:
            j = i
            while j < N - 1 and flat[j]:
                j += 1
            if j - i >= min_len:
                segments.append((i, j))
            i = j
        else:
            i += 1

    # Integrate step by step
    L = np.zeros_like(t)
    L[0] = L0
    seg_idx = 0

    i = 0
    while i < N - 1:
        # If this point is inside a plateau segment
        if seg_idx < len(segments) and segments[seg_idx][0] <= i < segments[seg_idx][1]:
            i_start, i_end = segments[seg_idx]
            k_plateau = np.mean(k_exp_t[i_start:i_end+1])
            dt_plateau = t[i_end] - t[i_start]
            Li = L[i_start]
            # Exact constant-k update
            L_next = (-D + np.sqrt(D**2 + 2*k_plateau*(D*Li + 0.5*k_plateau*Li**2 + A*D*k_plateau*dt_plateau))) / k_plateau
            L[i_start+1:i_end+1] = L_next
            i = i_end
            seg_idx += 1
        else:
            # Normal small-step exact update with local k
            k_mid = 0.5 * (k_exp_t[i] + k_exp_t[i+1])
            L[i+1] = (-D + np.sqrt(D**2 + 2*k_mid*(D*L[i] + 0.5*k_mid*L[i]**2 + A*D*k_mid*dt[i]))) / k_mid
            # L_anal = (-D_ec + np.sqrt((D_ec + k_exp * L0)**2 + 2 * A * D_ec * k_exp ** 2 * t)) / k_exp

            i += 1
    return L[-1]