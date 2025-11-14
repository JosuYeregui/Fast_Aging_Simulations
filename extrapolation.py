import pybamm
from matplotlib import pyplot as plt
import os
from copy import deepcopy
import time
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

def extract_y0_yend(sol):
    res = extract_y_from_solution(sol)
    return {k: v[0] for k, v in res.items()}, {k: v[-1] for k, v in res.items()}

def extract_y_from_solution(sol):

    Qp = sol["X-averaged positive particle concentration"].entries[-1] * \
         sol["Positive electrode capacity [A.h]"].entries
    Qn = sol["X-averaged negative particle concentration"].entries[-1] * \
         sol["Negative electrode capacity [A.h]"].entries
    nLi = 3600 / 96485.33 * (Qp + Qn)
    LLI_sei = sol["Loss of capacity to negative SEI [A.h]"].entries
    L_sei = sol["X-averaged negative SEI thickness [m]"].entries
    eps_n = sol["X-averaged negative electrode active material volume fraction"].entries
    eps_p = sol["X-averaged positive electrode active material volume fraction"].entries
    # eps_n = param["Negative electrode active material volume fraction"] * (1 - sol["Loss of active material in negative electrode [%]"].entries / 100.)
    # eps_p = param["Positive electrode active material volume fraction"] * (1 - sol["Loss of active material in positive electrode [%]"].entries / 100.)

    return {"nLi": nLi, "LLI_sei": LLI_sei, "L_sei": L_sei, "eps_n": eps_n, "eps_p": eps_p}

def extrapolate_solution(solution, n_cycles):

    vars_0, vars_end = extract_y0_yend(solution)

    extrapolated_vars = {}
    for k_0, v_0 in vars_0.items():
        v_end = vars_end[k_0]
        # y = mx+y0 = y0 + (y_end - y_0) / 1 * n_cycles
        extrapolated_vars[k_0] = v_end + (v_end - v_0) / 1 * (n_cycles - 1)

    # print(vars_end, extrapolated_vars)

    return extrapolated_vars

def load_extrapolated_into_solution(sol, extrapolated_vars):

    print("A")

    dQn = (sol["Negative electrode capacity [A.h]"].data[-1] * extrapolated_vars["eps_n"] /
           sol["X-averaged negative electrode active material volume fraction"].entries[-1])

    sol.last_state["X-averaged negative particle concentration [mol.m-3]"]._entries_raw =- extrapolated_vars["LLI_sei"] / dQn
    # sol["Negative SEI thickness [m]"].entries[-1] = extrapolated_vars["L_sei"]
    sol.last_state["X-averaged negative SEI thickness [m]"]._entries_raw = [extrapolated_vars["L_sei"]]
    # theta_n_berria = theta_n - n_LLI_SEI / Q_n ?

    # Load the extrapolated eps
    sol.last_state["X-averaged negative electrode active material volume fraction"]._entries_raw =  [extrapolated_vars["eps_n"]]
    sol.last_state["X-averaged positive electrode active material volume fraction"]._entries_raw =  [extrapolated_vars["eps_p"]]

    return sol

import numpy as np
def update_input_params(sol, param_input, extrapolated_vars):

    dQn = (sol["Negative electrode capacity [A.h]"].entries[-1] * extrapolated_vars["eps_n"] /
           sol["X-averaged negative electrode active material volume fraction"].entries[-1])

    #param_input = {}
    param_input["Initial concentration in negative electrode [mol.m-3]"] -= extrapolated_vars["LLI_sei"] / dQn * parameter_values["Maximum concentration in negative electrode [mol.m-3]"]

    #param_input["Initial concentration in negative electrode [mol.m-3]"] = (
    #        sol["X-averaged negative particle concentration [mol.m-3]"].entries[0, -1] -
    #        extrapolated_vars["LLI_sei"] / dQn)
    # sol["Negative SEI thickness [m]"].entries[-1] = extrapolated_vars["L_sei"]
    param_input["Initial SEI thickness [m]"] = extrapolated_vars["L_sei"]
    # theta_n_berria = theta_n - n_LLI_SEI / Q_n ?

    # Load the extrapolated eps
    param_input["Negative electrode active material volume fraction"] =  extrapolated_vars["eps_n"]
    param_input["Positive electrode active material volume fraction"] =  extrapolated_vars["eps_p"]

    return param_input

# sol_baseline.cycles[-1].steps[-1].y_slices
if __name__ == "__main__":

    tot_cycles = 1000 # Total cycles to simulate
    n_cycles = 25 # each how many cycles to extrapolate

    experiment_baseline = pybamm.Experiment(
        [(
                "Discharge at 1C until 3V",
                "Rest for 1 hour",
                "Charge at 1C until 4.2V",
                "Hold at 4.2V until C/50",)
        ] * tot_cycles
    )
    #parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values = pybamm.ParameterValues("Mohtat2020")
    parameter_values.update({"SEI kinetic rate constant [m.s-1]": 1e-15})
    parameter_values.update({"Negative electrode LAM constant proportional term [s-1]": 1.7e-7}, check_already_exists=False)  # Sulzer 2021 std
    parameter_values.update({"Positive electrode LAM constant proportional term [s-1]": 1.7e-7}, check_already_exists=False)

    param_Ai = pybamm.ParameterValues("Ai2020")
    parameters_to_change = ["Negative electrode partial molar volume [m3.mol-1]",
                            "Positive electrode partial molar volume [m3.mol-1]",
                            "Negative electrode LAM constant exponential term",
                            "Positive electrode LAM constant exponential term",
                            "Negative electrode LAM constant proportional term [s-1]",
                            "Positive electrode LAM constant proportional term [s-1]",
                            "Negative electrode critical stress [Pa]",
                            "Positive electrode critical stress [Pa]",
                            "Negative electrode Young's modulus [Pa]",
                            "Positive electrode Young's modulus [Pa]",
                            "Negative electrode Poisson's ratio",
                            "Positive electrode Poisson's ratio",
                            "Negative electrode reference concentration for free of deformation [mol.m-3]",
                            "Positive electrode reference concentration for free of deformation [mol.m-3]",
                            "Negative electrode volume change",
                            "Positive electrode volume change",
                            "Cell thermal expansion coefficient [m.K-1]"]
    for p in parameters_to_change:
        parameter_values.update({p: param_Ai[p]}, check_already_exists=False)
        # print(p, "updated to ", str(param_Ai[p]))

    #parameter_values.update({"Negative electrode partial molar volume [m3.mol-1]": param_Ai[]}, check_already_exists=False)
    #parameter_values.update({"Positive electrode partial molar volume [m3.mol-1]": 0.}, check_already_exists=False)

    parameter_values.set_initial_stoichiometries(1)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values)

    spm = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited", "loss of active material": "stress-driven","x-average side reactions":"true"})
    # spm = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"})
    sim_baseline = pybamm.Simulation(spm, experiment=experiment_baseline, parameter_values=parameter_values)
    start_time = time.time()
    sol_baseline = sim_baseline.solve()
    end_time = time.time()
    print("Baseline simulation time: ", end_time - start_time)

    Q_bs = [get_capacity_from_sol(s, parameter_values, 0, esoh_solver) for s in sol_baseline.cycles]
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
    starting_sol = None

    param_input = {"Initial SEI thickness [m]": parameter_values["Initial SEI thickness [m]"],
                   "Negative electrode active material volume fraction": parameter_values[
                       "Negative electrode active material volume fraction"],
                   "Positive electrode active material volume fraction": parameter_values[
                       "Positive electrode active material volume fraction"],
                   "Initial concentration in negative electrode [mol.m-3]": parameter_values[
                       "Initial concentration in negative electrode [mol.m-3]"]}
    cs_ini = param_input["Initial concentration in negative electrode [mol.m-3]"]


    parameter_values.update({"Initial SEI thickness [m]": "[input]",
                             "Negative electrode active material volume fraction": "[input]",
                             "Positive electrode active material volume fraction": "[input]",
                             "Initial concentration in negative electrode [mol.m-3]": "[input]"})


    sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=parameter_values)

    extrap_time = 0.
    for i in range(int(tot_cycles//n_cycles) + 1):
        # Simulate the experiment from start or (modified) previous solution
        start_extrap_time = time.time()
        if starting_sol is not None:
            sol = sim.solve(inputs=param_input, starting_solution=None)#starting_solution=starting_sol)
        else:
            sol = sim.solve(inputs=param_input)
        extrap_time += time.time() - start_extrap_time
        # Last cycle containing the steps of the defined experiment sequence
        sol_last = sol.cycles[-1]
        # Get the cycle output information
        extrapolation_results["cycle"].append(i*n_cycles)
        extrapolation_results["Q"].append(get_capacity_from_sol(sol_last, parameter_values, 0, esoh_solver))
        #extrapolation_results["eps_n"].append(sol_last["X-averaged negative electrode active material volume fraction"].entries[-1])
        extrapolation_results["eps_n"].append(
            sol_last["Negative electrode capacity [A.h]"].entries[-1])

        # Extrapolate variables using the last cycle simulation information
        extrapolated_vars = extrapolate_solution(sol_last, n_cycles)

        # Hemen parameter load-a egin behar dugu
        #sol_last = load_extrapolated_into_solution(sol_last, extrapolated_vars)
        param_input = update_input_params(sol_last, param_input, extrapolated_vars)
        #sim.solution = None

        #starting_sol = sol_last

    print("Extrapolation time: ", extrap_time)

    fig, ax = plt.subplots()
    ax.plot(Q_bs, label="Baseline")
    ax.scatter(extrapolation_results["cycle"], extrapolation_results["Q"], label="Extrapolated N=" + str(n_cycles),
               c="red")
    #ax.scatter(list(range(len(extrapolation_results["Q"]))), extrapolation_results["Q"], label="Extrapolated N=1",
    #           c="green")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Discharge capacity [A.h]")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(eps_bs, label="Baseline")
    ax.scatter(extrapolation_results["cycle"], extrapolation_results["eps_n"], label="Extrapolated N=" + str(n_cycles),
               c="red")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Discharge capacity [A.h]")
    plt.legend()
    plt.show()

