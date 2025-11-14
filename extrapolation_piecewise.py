import pybamm
from matplotlib import pyplot as plt
import os
import time
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_capacity_from_sol(solution, param, idx, esoh_solver):
    x_0, _, x_100, _ = extract_stoich_from_sol(solution, idx, esoh_solver)
    eps_n = solution["X-averaged negative electrode active material volume fraction"].entries[idx]
    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    return eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600

def extract_stoich_from_sol(solution, idx, esoh_solver):

    Qn = solution["Negative electrode capacity [A.h]"].entries[idx]
    Qp = solution["Positive electrode capacity [A.h]"].entries[idx]
    Qp_0 = solution["X-averaged positive particle concentration"].entries[-1][idx] * \
           solution["Positive electrode capacity [A.h]"].entries[idx]
    Qn_0 = solution["X-averaged negative particle concentration"].entries[-1][idx] * \
           solution["Negative electrode capacity [A.h]"].entries[idx]
    QLi = (Qp_0 + Qn_0)

    inputs = {"V_min": 3, "V_max": 4.2,
              "Q_n": Qn,
              "Q_p": Qp,
              "Q_Li": QLi}

    esoh_sol = esoh_solver.solve(inputs)

    x_0 = esoh_sol["x_0"]
    y_0 = esoh_sol["y_0"]
    x_100 = esoh_sol["x_100"]
    y_100 = esoh_sol["y_100"]

    return x_0, y_0, x_100, y_100

def extract_y0_yend(sol, Lsei):
    res = extract_y(sol, Lsei)
    return {k: v[0] for k, v in res.items()}, {k: v[-1] for k, v in res.items()}

def extract_y(sol, Lsei):

    Qp = sol["X-averaged positive particle concentration"].entries[-1] * \
         sol["Positive electrode capacity [A.h]"].entries
    Qn = sol["X-averaged negative particle concentration"].entries[-1] * \
         sol["Negative electrode capacity [A.h]"].entries
    nLi = 3600 / 96485.33 * (Qp + Qn)
    LLI_sei = sol["Loss of capacity to negative SEI [A.h]"].entries
    L_sei = Lsei
    # L_sei = sol["X-averaged negative SEI thickness [m]"].entries
    eps_n = sol["X-averaged negative electrode active material volume fraction"].entries
    eps_p = sol["X-averaged positive electrode active material volume fraction"].entries
    # eps_n = param["Negative electrode active material volume fraction"] * (1 - sol["Loss of active material in negative electrode [%]"].entries / 100.)
    # eps_p = param["Positive electrode active material volume fraction"] * (1 - sol["Loss of active material in positive electrode [%]"].entries / 100.)

    return {"nLi": nLi, "LLI_sei": LLI_sei, "L_sei": L_sei, "eps_n": eps_n, "eps_p": eps_p}

def extrapolate_solution(solution, Lsei, n_cycles):

    vars_0, vars_end = extract_y0_yend(solution, Lsei)

    extrapolated_vars = {}
    for k_0, v_0 in vars_0.items():
        v_end = vars_end[k_0]
        # y = mx+y0 = y_end + (y_end - y_0) / 1 * (n_cycles - 1)
        extrapolated_vars[k_0] = v_end + (v_end - v_0) / 1 * (n_cycles - 1)

    # print(vars_end, extrapolated_vars)

    return extrapolated_vars

def load_extrapolated_into_params(param, sol, extrapolated_vars):

    dQn = (sol["Negative electrode capacity [A.h]"].entries[-1] * extrapolated_vars["eps_n"] /
           sol["X-averaged negative electrode active material volume fraction"].entries[-1])

    param["Initial concentration in negative electrode [mol.m-3]"] -= extrapolated_vars["LLI_sei"] / dQn * param["Maximum concentration in negative electrode [mol.m-3]"]
    # sol["Negative SEI thickness [m]"].entries[-1] = extrapolated_vars["L_sei"]
    param["Initial SEI thickness [m]"] = extrapolated_vars["L_sei"]
    # theta_n_berria = theta_n - n_LLI_SEI / Q_n ?

    # Load the extrapolated eps
    param["Negative electrode active material volume fraction"] = extrapolated_vars["eps_n"]
    param["Positive electrode active material volume fraction"] = extrapolated_vars["eps_p"]

    param.set_initial_stoichiometries(1)

    return param

def piecewise_integration(t, k_exp_t, D, A, L0, slope_tol=1e-3, min_len=5):
    """
    Piecewise integration for dL/dt = A*D*k/(D + k*L)
    treating nearly-constant k segments analytically.
    """
    t = np.asarray(t)
    k_exp_t = np.asarray(k_exp_t)
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
    return L, flat, segments

def perform_piecewise_integration(sol, param):

    j0 = float(param.get("SEI reaction exchange current density [A.m-2]",
                         1.5e-7))  # or "SEI reaction exchange current density"
    k_sei = float(param.get("SEI kinetic rate constant [m.s-1]", 1e-12))
    D_ec = float(param.get("EC diffusivity [m2.s-1]", 2e-18))  # or "Molar mass of SEI"
    c_ec = float(param.get("EC initial concentration in electrolyte [mol.m-3]", 4541))
    alpha = float(param.get("SEI charge transfer coefficient", 0.5))
    U_sei = float(param.get("SEI open-circuit potential [V]", 0.4))
    F = float(param.get("Faraday constant [C.mol-1]", 96485.33212))
    R = float(param.get("Ideal gas constant [J.mol-1.K-1]", 8.314462618))
    T = float(param.get("Ambient temperature [K]", 298.15))
    L0 = float(param.get("Initial SEI thickness [m]", 5e-9))
    V_sei = float(param.get("SEI partial molar volume [m3.mol-1]", 9.585e-5))

    t = sol["Time [s]"].entries

    eta = sol["X-averaged negative electrode open-circuit potential [V]"](t) - U_sei

    exp_vals = np.exp(-alpha * F * eta / (R * T))
    k_exp = exp_vals * k_sei
    A = c_ec * V_sei / 2

    L_anal, _, _ = piecewise_integration(t, k_exp, D_ec, A, L0)
    return L_anal

# sol_baseline.cycles[-1].steps[-1].y_slices
if __name__ == "__main__":

    tot_cycles = 300 # Total cycles to simulate
    n_cycles = 10 # each how many cycles to extrapolate

    experiment_baseline = pybamm.Experiment(
        [(
                "Discharge at 1C until 3V",
                "Rest for 1 hour",
                "Charge at 1C until 4.2V",
                "Hold at 4.2V until C/50",)
        ] * tot_cycles
    )

    param = pybamm.ParameterValues("Chen2020")

    # parameter_values = pybamm.ParameterValues("Mohtat2020")
    # parameter_values.update({"SEI kinetic rate constant [m.s-1]": 1e-15})
    # parameter_values.update({"Negative electrode LAM constant proportional term [s-1]": 1.7e-7}, check_already_exists=False)  # Sulzer 2021 std
    # parameter_values.update({"Positive electrode LAM constant proportional term [s-1]": 1.7e-7}, check_already_exists=False)
    #
    # param_Ai = pybamm.ParameterValues("Ai2020")
    # parameters_to_change = ["Negative electrode partial molar volume [m3.mol-1]",
    #                         "Positive electrode partial molar volume [m3.mol-1]",
    #                         "Negative electrode LAM constant exponential term",
    #                         "Positive electrode LAM constant exponential term",
    #                         "Negative electrode LAM constant proportional term [s-1]",
    #                         "Positive electrode LAM constant proportional term [s-1]",
    #                         "Negative electrode critical stress [Pa]",
    #                         "Positive electrode critical stress [Pa]",
    #                         "Negative electrode Young's modulus [Pa]",
    #                         "Positive electrode Young's modulus [Pa]",
    #                         "Negative electrode Poisson's ratio",
    #                         "Positive electrode Poisson's ratio",
    #                         "Negative electrode reference concentration for free of deformation [mol.m-3]",
    #                         "Positive electrode reference concentration for free of deformation [mol.m-3]",
    #                         "Negative electrode volume change",
    #                         "Positive electrode volume change",
    #                         "Cell thermal expansion coefficient [m.K-1]"]
    # for p in parameters_to_change:
    #     parameter_values.update({p: param_Ai[p]}, check_already_exists=False)
    #     # print(p, "updated to ", str(param_Ai[p]))

    #parameter_values.update({"Negative electrode partial molar volume [m3.mol-1]": param_Ai[]}, check_already_exists=False)
    #parameter_values.update({"Positive electrode partial molar volume [m3.mol-1]": 0.}, check_already_exists=False)

    param.set_initial_stoichiometries(1)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    spm = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited","x-average side reactions":"false"})
    sim_baseline = pybamm.Simulation(spm, experiment=experiment_baseline, parameter_values=param)
    start_time = time.time()
    sol_baseline = sim_baseline.solve()
    end_time = time.time()
    print("Baseline simulation time: ", end_time - start_time)

    Q_bs = [get_capacity_from_sol(s, param, 0, esoh_solver) for s in sol_baseline.cycles]
    # eps_bs = [s["X-averaged negative electrode active material volume fraction"].entries[-1] for s in sol_baseline.cycles]
    eps_bs = [s["Negative electrode capacity [A.h]"].entries[-1] for s in
              sol_baseline.cycles]

    experiment = pybamm.Experiment(
        [(
            "Discharge at 1C until 3V",
            "Rest for 1 hour",
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/50",)
        ]
    )

    extrapolation_results = {"cycle": [], "Q": [], "eps_n": [], "eps_p": [], "L_sei": [], "LLI_sei": [], "nLi":[]}

    extrap_time = 0.
    for i in range(int(tot_cycles//n_cycles) + 1):

        sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=param)
        # Simulate the experiment from start or (modified) previous solution
        start_extrap_time = time.time()
        sol = sim.solve()
        extrap_time += time.time() - start_extrap_time
        # Last cycle containing the steps of the defined experiment sequence
        sol_last = sol.cycles[-1]
        # Get the cycle output information
        extrapolation_results["cycle"].append(i*n_cycles)
        extrapolation_results["Q"].append(get_capacity_from_sol(sol_last, param, 0, esoh_solver))
        #extrapolation_results["eps_n"].append(sol_last["X-averaged negative electrode active material volume fraction"].entries[-1])
        extrapolation_results["eps_n"].append(
            sol_last["Negative electrode capacity [A.h]"].entries[-1])

        # Extrapolate variables using the last cycle simulation information
        Lsei = perform_piecewise_integration(sol_last, param)
        extrapolated_vars = extrapolate_solution(sol_last, Lsei, n_cycles)

        # Hemen parameter load-a egin behar dugu
        param = load_extrapolated_into_params(param, sol_last, extrapolated_vars)

    print("Extrapolation + Piecewise time: ", extrap_time)

    fig, ax = plt.subplots()
    ax.plot(Q_bs, label="Baseline")
    ax.scatter(extrapolation_results["cycle"], extrapolation_results["Q"], label="Extrapolated + pw N=" + str(n_cycles),
               c="red")
    #ax.scatter(list(range(len(extrapolation_results["Q"]))), extrapolation_results["Q"], label="Extrapolated N=1",
    #           c="green")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Discharge capacity [A.h]")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(eps_bs, label="Baseline")
    ax.scatter(extrapolation_results["cycle"], extrapolation_results["eps_n"], label="Extrapolated + pw N=" + str(n_cycles),
               c="red")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Discharge capacity [A.h]")
    plt.legend()
    #plt.show()

