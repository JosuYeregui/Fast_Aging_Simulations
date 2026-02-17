import numpy as np
from utils.characterize import get_Q, get_Qn, stoichiometric_window

def get_dif(y, t, cycler, input_vars):

    L0 = y[0]

    L = tfunc_integration_LSEI(L0, t, cycler, input_vars)
    dLdt = (L - L0)
    # L = piecewise_integration_LSEI(L0, cycler, input_vars)
    # dLdt = (L - L0) / cycler.total_time

    deps_dt_n = stressLAM_deps_dt(t, y[1], cycler, input_vars, "n")

    deps_dt_p = stressLAM_deps_dt(t, y[2], cycler, input_vars, "p")

    dPl_dt = dPlating_dt(t, y, cycler, input_vars)

    return [dLdt, deps_dt_n, deps_dt_p, dPl_dt]


def stressLAM_deps_dt(t, eps, cycler, input_vars, domain):
    # I, _ = cycler(t/3600.)
    I_all = np.asarray([s.I_avg - s.I_avg_dyn[domain] for s in cycler.steps])
    if domain == "p":
        I_all[I_all > 0] = 0
    else:
        I_all[I_all < 0] = 0
    sign = np.sign(np.mean(I_all))
    I_all = np.abs(I_all)
    m = input_vars["m_" + domain]
    t_steps = np.diff(cycler.times)
    num = [I ** m  * t_steps[idx] for idx, I in enumerate(I_all)]
    I_avg = sign * (np.sum(num) / cycler.times[-1]) ** (1/m)

    deps_dt = []

    cs_cave = input_vars["cs_cave_"+domain](I_avg, eps)
    Omega = input_vars["Omega_"+domain]
    E = input_vars["E_"+domain]
    nu = input_vars["nu_"+domain]
    stress_geometric_factor = input_vars["omega_fact_"+domain]

    stress_t = Omega * E * cs_cave / (1 - nu) / stress_geometric_factor
    stress_r = 0.

    stress_h = (stress_r + 2 * stress_t) / 3

    if stress_h <= 0:
        deps_dt.append(0.)
    else:

        stress_critial = input_vars["stress_critical_"+domain]
        Beta = input_vars["Beta_"+domain]

        deps_dt.append(-Beta * np.abs(stress_h / stress_critial) ** m)

    return np.sum(deps_dt * t_steps) / np.sum(t_steps)


def tfunc_integration_LSEI(L0, t, cycler, input_vars):
    """
    Integration for dL/dt = A*D*k/(D + k*L)
    on a single time step.
    """
    k_exp_t = input_vars["k_exp"]
    D_ec = input_vars["D_ec"]
    A_ec = input_vars["A_ec"]

    dt = 1.

    _, soc = cycler(t/3600.)
    _, soc_dt = cycler((t + dt)/3600.)

    # Normal small-step exact update with local k
    k_mid = 0.5 * (k_exp_t(soc) + k_exp_t(soc_dt))
    L= (-D_ec + np.sqrt(D_ec**2 + 2*k_mid*(D_ec*L0 + 0.5*k_mid*L0**2 + A_ec*D_ec*k_mid*dt))) / k_mid
    return L


def piecewise_integration_LSEI(L0, cycler, input_vars, slope_tol=1e-3, min_len=5):
    """
    Piecewise integration for dL/dt = A*D*k/(D + k*L)
    treating nearly-constant k segments analytically.
    """
    k_exp_fn = input_vars["k_exp"]
    D_ec = input_vars["D_ec"]
    A_ec = input_vars["A_ec"]

    t = np.arange(0, cycler.total_time, 10/3600)

    k_exp_t = []
    for t_ in t:
        _, soc_ = cycler(t_)
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
            L_next = (-D_ec + np.sqrt(D_ec**2 + 2*k_plateau*(D_ec*Li + 0.5*k_plateau*Li**2 + A_ec*D_ec*k_plateau*dt_plateau))) / k_plateau
            L[i_start+1:i_end+1] = L_next
            i = i_end
            seg_idx += 1
        else:
            # Normal small-step exact update with local k
            k_mid = 0.5 * (k_exp_t[i] + k_exp_t[i+1])
            L[i+1] = (-D_ec + np.sqrt(D_ec**2 + 2*k_mid*(D_ec*L[i] + 0.5*k_mid*L[i]**2 + A_ec*D_ec*k_mid*dt[i]))) / k_mid
            # L_anal = (-D_ec + np.sqrt((D_ec + k_exp * L0)**2 + 2 * A * D_ec * k_exp ** 2 * t)) / k_exp

            i += 1
    return L[-1]

def dPlating_dt(t, y, cycler, input_vars):
    I, soc = cycler(t/3600.)
    I = -I / input_vars["A"]

    as_n = 3 * y[1] / input_vars["R_n"]  # specific surface area of the negative electrode

    ce_int = (input_vars["ce_0"] + I * (1- input_vars["t_plus"])/input_vars["F"] *
              (input_vars["L_n"] / (3* input_vars["Deff_n"])- input_vars["L_s"]/ (2* input_vars["Deff_s"])))
    ce_int =  ce_int /input_vars["ce_0"]

    sol_n = soc * (input_vars["x_100"] - input_vars["x_0"]) + input_vars["x_0"]
    j0 = np.sqrt(ce_int) * np.sign(sol_n) * np.sqrt(1 - sol_n) * input_vars["m_n"] * input_vars["F"]
    # Compute plating rate based on current and available lithium
    RTF = input_vars["R"]*input_vars["T"]/input_vars["F"]
    eta_r = -2 * RTF * np.asinh(I/(as_n * j0 * input_vars["L_n"]))

    ocp = input_vars["U_n"](sol_n)
    eta_plating = -(eta_r + ocp)

    plating_rate = input_vars["k_plating"] * np.exp(input_vars["alpha_plating"] * eta_plating / RTF)
    return plating_rate