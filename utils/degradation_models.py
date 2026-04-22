import numpy as np
from utils.characterize import get_Q, get_Qn, stoichiometric_window

def get_dif(y, t, cycler, input_vars):

    L0 = y[0]

    #L = tfunc_integration_LSEI(L0, y[1], t, cycler, input_vars)
    #dLdt = (L - L0)
    L = lliaware_integration_LSEI(L0, y[1], cycler, input_vars)
    dLdt = (L - L0) / (cycler.total_time)

    deps_dt_n = stressLAM_deps_dt(t, y[1], cycler, input_vars, "n")

    deps_dt_p = stressLAM_deps_dt(t, y[2], cycler, input_vars, "p")

    #dPl_dt = dPlating_dt(t, y, cycler, input_vars)
    L_Pl = piecewise_integration_Plating(y, cycler, input_vars)
    dPl_dt = L_Pl

    return [dLdt, deps_dt_n, deps_dt_p, dPl_dt]

def get_t_and_dt(cycler):
    times = np.array(cycler.times)

    points_per_step = 10
    t_segments = []

    for i in range(len(times) - 1):
        start_time = times[i]
        end_time = times[i + 1]
        is_last_step = (i == len(times) - 2)
        step_t = np.linspace(start_time, end_time, points_per_step, endpoint=is_last_step)
        t_segments.append(step_t)

    t = np.concatenate(t_segments)

    dt_array = np.diff(t)
    dt_array = np.append(dt_array, dt_array[-1])

    return t, dt_array


def stressLAM_deps_dt(t, eps, cycler, input_vars, domain):

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
    I_avg = sign * (np.sum(num)/cycler.times[-1]) ** (1/m)

    cs_cave = input_vars["cs_cave_"+domain](I_avg, eps)
    Omega = input_vars["Omega_"+domain]
    E = input_vars["E_"+domain]
    nu = input_vars["nu_"+domain]
    stress_geometric_factor = input_vars["omega_fact_"+domain]

    stress_t = Omega * E * cs_cave / (1 - nu) / stress_geometric_factor

    stress_h = (2 * stress_t) / 3

    stress_critial = input_vars["stress_critical_"+domain]
    Beta = input_vars["Beta_"+domain]

    deps_dt = -Beta * np.abs(stress_h / stress_critial) ** m

    return deps_dt


def tfunc_integration_LSEI(L0, eps_n, t, cycler, input_vars):
    """
    Integration for dL/dt = A*D*k/(D + k*L)
    on a single time step.
    """

    def k_exp(soc): # TODO: Sieso sartukoziou csurf-ekiko cave beharrean
        x_0 = input_vars["x_0"]
        x_100 = input_vars["x_100"]
        sol_n = soc * (x_100 - x_0) + x_0
        eta = input_vars["U_n"](sol_n) - input_vars["U_sei"]
        arrhenius = np.exp(input_vars["E_act_SEI"] / input_vars["R"]* (1 / 298.15 - 1 / input_vars["T"]))
        k_sei = input_vars["k_sei"] * arrhenius
        return np.exp(
            -input_vars["alpha"] * input_vars["F"] * eta /
            (input_vars["R"] * input_vars["T"])
        ) * k_sei

    k_exp_t = k_exp
    D_ec = input_vars["D_ec"]
    A_ec = input_vars["A_ec"]

    dt = 1.

    _, soc = cycler(t/3600.)
    _, soc_dt = cycler((t + dt)/3600.)

    Qn = get_Qn(input_vars, eps_n)
    F = 96485.33
    A = input_vars["A"]
    a_s = eps_n * 3 / input_vars["R_n"]
    Ln = input_vars["L_n"]
    LLI_SEI_stress = (L0 - input_vars["L_0"]) * A * F / (input_vars["V_SEI"] * 3600.) * input_vars["z_SEI"] * a_s * Ln

    soc_comp = LLI_SEI_stress/Qn
    #soc_comp = (sol_comp - input_vars["x_0"]) / (input_vars["x_100"] - input_vars["x_0"])

    # Normal small-step exact update with local k
    k_mid = 0.5 * (k_exp_t(max(soc - soc_comp, 0.)) + k_exp_t(max(soc_dt - soc_comp, 0.)))
    L = (-D_ec + np.sqrt(D_ec**2 + 2*k_mid*(D_ec*L0 + 0.5*k_mid*L0**2 + A_ec*D_ec*k_mid*dt))) / k_mid
    return L


def lliaware_integration_LSEI(L0, eps_n, cycler, input_vars):
    """
    Integration for dL/dt = A*D*k/(D + k*L)
    on a single time step.
    """
    D_ec = input_vars["D_ec"]
    A_ec = input_vars["A_ec"]

    # t = np.linspace(0, cycler.total_time, 1000)
    # dt = cycler.total_time / 1000.

    t, dt_array = get_t_and_dt(cycler)

    Qn = get_Qn(input_vars, eps_n)
    F = 96485.33
    A = input_vars["A"]
    a_s = eps_n * 3 / input_vars["R_n"]
    Ln = input_vars["L_n"]

    #sol_n = input_vars["cini_n"] / input_vars["csmax_n"]

    L = L0

    soc_vec = []
    _, soc = cycler(0.)
    soc_prev = soc
    soc_chg = 0.

    for t_ in t:
        _, soc = cycler(t_)
        soc_vec.append(soc)
        soc_dif = soc - soc_prev
        soc_prev = soc
        soc_chg += max(soc_dif, 0)

    for t_, soc, dt in zip(t, soc_vec, dt_array):

        LLI_SEI_stress = (L - input_vars["L_0"]) * A * F / (input_vars["V_SEI"] * 3600.) * input_vars["z_SEI"] * a_s * Ln

        LLI = LLI_SEI_stress  # + LLI_pl

        # Update parameters
        soc_comp = max(LLI / Qn - soc_chg, 0)
        #sol_n = input_vars["cini_n"] / input_vars["csmax_n"] - LLI / Qn
        #soc_n = (sol_n - input_vars["x_0"]) / (input_vars["x_100"] - input_vars["x_0"])
        sol_n = max(soc-soc_comp, 0) * (input_vars["x_100"] - input_vars["x_0"]) + input_vars["x_0"]

        eta = input_vars["U_n"](sol_n) - input_vars["U_sei"]
        arrhenius = np.exp(input_vars["E_act_SEI"] / input_vars["R"] * (1 / 298.15 - 1 / input_vars["T"]))
        k_sei = input_vars["k_sei"] * arrhenius
        k_mid = np.exp(
            -input_vars["alpha"] * input_vars["F"] * eta /
            (input_vars["R"] * input_vars["T"])
        ) * k_sei
        L = (-D_ec + np.sqrt(
            D_ec ** 2 + 2 * k_mid * (D_ec * L + 0.5 * k_mid * L ** 2 + A_ec * D_ec * k_mid * dt))) / k_mid

    return L


def dPlating_dt(t, y, cycler, input_vars):
    I_raw, soc = cycler(t/3600.)
    I_raw = -I_raw

    A = input_vars["A"]
    F = input_vars["F"]
    R = input_vars["R"]
    T = input_vars["T"]
    RTF = R * T / F

    I = I_raw / A

    as_n = 3 * y[1] / input_vars["R_n"]  # specific surface area of the negative electrode

    # ce_int = (input_vars["ce_0"] + I * (1 - input_vars["t_plus"]) / F *
    #           (input_vars["L_n"] / (3 * input_vars["Deff_n"]) - input_vars["L_s"] / (2 * input_vars["Deff_s"])))
    ce_int = (input_vars["ce_0"] + I * (1 - input_vars["t_plus"]) / F *
              (input_vars["L_n"] / (3 * input_vars["Deff_n"])))
    # ce_int =  ce_int

    sol_n = soc * (input_vars["x_100"] - input_vars["x_0"]) + input_vars["x_0"]
    sol_n += input_vars["cs_cave_n"](I_raw, y[1]) / input_vars["csmax_n"]  # We need surface concentration

    j0 = np.sqrt(ce_int) * np.sqrt(sol_n * input_vars["csmax_n"]) * np.sqrt(
        input_vars["csmax_n"] - sol_n * input_vars["csmax_n"]) * input_vars["m_ref"]
    # j0 = np.maximum(j0, 1e-10)  # Avoid division by zero in asinh

    eta_r = -2 * RTF * np.arcsinh(I / (as_n * j0 * input_vars["L_n"]))

    ocp = input_vars["U_n"](sol_n)
    eta_plating = -(eta_r - ocp)
    i0 = input_vars["k_plating"] * ce_int * F
    # i0 = 1e-4*ce_int
    j_k = i0 * np.exp(-input_vars["alpha_plating"] * eta_plating / RTF)
    return j_k * input_vars["V_Li"] / F


def piecewise_integration_Plating(y, cycler, input_vars, slope_tol=1e-15, min_len=10):
    """
    Piecewise integration for Lithium Plating.

    1. Pre-calculates the 'instantaneous plating rate' across the cycle.
    2. Detects 'plateaus' where the rate is nearly constant.
    3. Integrates fast over plateaus and step-by-step over transients.
    """

    #t = np.linspace(0, cycler.total_time, 1000)
    t, _ = get_t_and_dt(cycler)

    I_t = []
    soc_t = []

    for t_i in t:
        i_val, s_val = cycler(t_i)
        I_t.append(-i_val)
        soc_t.append(s_val)

    I_raw = np.array(I_t)
    soc_t = np.array(soc_t)

    A = input_vars["A"]
    F = input_vars["F"]
    R = input_vars["R"]
    T = input_vars["T"]
    RTF = R * T / F

    as_n = 3 * y[1] / input_vars["R_n"]
    I = I_raw / A

    ce_int = (input_vars["ce_0"] + I * (1 - input_vars["t_plus"]) / F *
              (input_vars["L_n"] / (3 * input_vars["Deff_n"])))

    # ce_int = np.maximum(ce_int, 1e-10)

    sol_n = soc_t * (input_vars["x_100"] - input_vars["x_0"]) + input_vars["x_0"]
    sol_n += input_vars["cs_cave_n"](I_raw, y[1]) / input_vars["csmax_n"]  # We need surface concentration

    j0 = np.sqrt(ce_int) * np.sqrt(sol_n * input_vars["csmax_n"]) * np.sqrt(input_vars["csmax_n"] - sol_n * input_vars["csmax_n"]) * input_vars["m_ref"]
    # j0 = np.maximum(j0, 1e-10)  # Avoid division by zero in asinh

    eta_r = -2 * RTF * np.arcsinh(I / (as_n * j0 * input_vars["L_n"]))

    U_n_vec = np.vectorize(input_vars["U_n"])
    ocp = U_n_vec(sol_n)

    eta_plating = -(eta_r - ocp)
    i0 = input_vars["k_plating"] * ce_int * F
    #i0 = 1e-4*ce_int
    j_k = i0 * np.exp(-input_vars["alpha_plating"] * eta_plating / RTF)

    rate_t = j_k#np.gradient(j_k, t)

    flat = np.abs(rate_t) < slope_tol

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

    L_plated = 0.0  # Initial capacity
    seg_idx = 0
    i = 0

    while i < N - 1:
        if seg_idx < len(segments) and segments[seg_idx][0] <= i < segments[seg_idx][1]:
            i_start, i_end = segments[seg_idx]

            rate_plateau = np.mean(rate_t[i_start:i_end + 1])
            dt_plateau = t[i_end] - t[i_start]

            L_plated += rate_plateau * dt_plateau * input_vars["V_Li"] / F

            i = i_end
            seg_idx += 1
        else:

            # Trapezoidal method
            dt = t[i + 1] - t[i]
            rate_step = 0.5 * (rate_t[i] + rate_t[i + 1])

            L_plated += rate_step * dt * input_vars["V_Li"] / F
            i += 1

    return L_plated / cycler.total_time