import numpy as np

def get_dif(y, t, cycler, input_vars):

    L0 = y[0]

    L = tfunc_integration_LSEI(L0, t, cycler, input_vars)
    dLdt = (L - L0)
    # L = piecewise_integration_LSEI(L0, cycler, input_vars)
    # dLdt = (L - L0) / cycler.total_time

    deps_dt_n = stressLAM_deps_dt(t, y[1], cycler, input_vars, "n")

    deps_dt_p = stressLAM_deps_dt(t, y[2], cycler, input_vars, "p")

    # t_array = np.linspace(t, t + cycler.total_time*3600., 100)
    # deps_dt_n = []
    # deps_dt_p = []
    # for t_ in t_array:
    #     deps_dt_n.append(stressLAM_deps_dt(t_, y[1], cycler, input_vars, "n"))
    #
    #     deps_dt_p.append(stressLAM_deps_dt(t_, y[2], cycler, input_vars, "p"))
    # deps_dt_n = np.mean(deps_dt_n)
    # deps_dt_p = np.mean(deps_dt_p)

    return [dLdt, deps_dt_n, deps_dt_p]

# def get_dif(y, t, cycler, input_vars):
#
#     L0 = y[0]
#
#     #L = tfunc_integration_LSEI(L0, t, cycler, input_vars)
#     L = piecewise_integration_LSEI(L0, cycler, input_vars)
#
#     return L - L0

def cast_params_for_LAM(param, input_vars):
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    F = 96485.33212

    names = {"Positive": "p", "Negative": "n"}
    constants = {"p": {}, "n": {}}
    for k, domain in names.items():
        input_vars["Omega_"+domain] = param[k + " electrode partial molar volume [m3.mol-1]"]
        input_vars["omega_fact_" + domain] = 3
        input_vars["E_"+domain] = param[k + " electrode Young\'s modulus [Pa]"]
        input_vars["nu_"+domain] = param[k + " electrode Poisson\'s ratio"]
        input_vars["stress_critical_"+domain] = param[k + " electrode critical stress [Pa]"]
        input_vars["Beta_"+domain] = param[k + " electrode LAM constant proportional term [s-1]"]
        input_vars["m_"+domain] = param[k + " electrode LAM constant exponential term"]

        D = param[k + " electrode diffusivity [m2.s-1]"]
        R = param[k + " particle radius [m]"]
        L = param[k + " electrode thickness [m]"]

        constants[domain]["D"] = D
        constants[domain]["R"] = R
        constants[domain]["L"] = L

        input_vars["Rpart_"+domain] = R

        def cs_cave(I, eps, domain):
            as_particle = eps * 3 / constants[domain]["R"]
            I = -I
            if domain == "n":
                j = I / (F * as_particle * constants[domain]["L"] * A)
            else:
                j = -I / (F * as_particle * constants[domain]["L"] * A)
            return -j / 5 * constants[domain]["R"] / constants[domain]["D"]

        input_vars["cs_cave_"+domain] = cs_cave
        input_vars["L_"+domain] = L
        input_vars["A"] = A
        input_vars["csmax_"+domain] = param["Maximum concentration in " + k.lower() + " electrode [mol.m-3]"]
        input_vars["cini_"+domain] = param["Initial concentration in " + k.lower() + " electrode [mol.m-3]"]
        input_vars["cini_" + domain + "_0"] = param["Initial concentration in " + k.lower() + " electrode [mol.m-3]"]
        input_vars["U_"+domain] = param[k + " electrode OCP [V]"]

    return input_vars

def stressLAM_deps_dt(t, eps, cycler, input_vars, domain):
    # I, _ = cycler(t/3600.)
    I_all = np.asarray([s.I_avg for s in cycler.steps])
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

    cs_cave = input_vars["cs_cave_"+domain](I_avg, eps, domain)
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

def cast_params_for_SEI(param, input_vars):

    k_sei = float(param.get("SEI kinetic rate constant [m.s-1]", 1e-12))
    D_ec = float(param.get("EC diffusivity [m2.s-1]", 2e-18))  # or "Molar mass of SEI"
    c_ec = float(param.get("EC initial concentration in electrolyte [mol.m-3]", 4541))
    alpha = float(param.get("SEI charge transfer coefficient", 0.5))
    U_sei = float(param.get("SEI open-circuit potential [V]", 0.4))
    F = float(param.get("Faraday constant [C.mol-1]", 96485.33212))
    R = float(param.get("Ideal gas constant [J.mol-1.K-1]", 8.314462618))
    T = float(param.get("Ambient temperature [K]", 298.15))
    V_sei = float(param.get("SEI partial molar volume [m3.mol-1]", 9.585e-5))
    z = float(param.get("Ratio of lithium moles to SEI moles", 2.))
    # x_0, x_100, _, _ = pybamm.lithium_ion.get_min_max_stoichiometries(param)
    OCP_n = param["Negative electrode OCP [V]"]
    rho_SEI = param["SEI resistivity [Ohm.m]"]

    def k_exp(soc):

        #sol_n = (soc - x_0) / (x_100 - x_0)
        x_0 = input_vars["x_0"]
        x_100 = input_vars["x_100"]
        sol_n = soc * (x_100 - x_0) + x_0

        #  = np.tile(SOL_n_cycle, n_cycles * 2)
        eta = OCP_n(sol_n) - U_sei

        exp_vals = np.exp(-alpha * F * eta / (R * T))
        k_exp_val = exp_vals * k_sei
        return k_exp_val

    A_ec = c_ec * V_sei / z

    input_vars["k_exp"] = k_exp
    input_vars["A_ec"] = A_ec
    input_vars["D_ec"] = D_ec
    input_vars["rho_SEI"] = rho_SEI
    input_vars["V_SEI"] = V_sei
    input_vars["z_SEI"] = z

    return input_vars

def tfunc_integration_LSEI(L0, t, cycler, input_vars):
    """
    Piecewise integration for dL/dt = A*D*k/(D + k*L)
    treating nearly-constant k segments analytically.
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