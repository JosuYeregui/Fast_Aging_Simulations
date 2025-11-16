import pybamm
from matplotlib import pyplot as plt
import os
from copy import deepcopy
import time

from scipy.integrate import RK23
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

    LLI_sei = sol["Loss of capacity to negative SEI [A.h]"].entries
    L_sei = sol["X-averaged negative SEI thickness [m]"].entries
    eps_n = sol["X-averaged negative electrode active material volume fraction"].entries
    eps_p = sol["X-averaged positive electrode active material volume fraction"].entries

    return {"LLI_sei": LLI_sei, "L_sei": L_sei, "eps_n": eps_n, "eps_p": eps_p}

def extrapolate_solution(solution, n_cycles):

    vars_0, vars_end = extract_y0_yend(solution)

    extrapolated_vars = {}
    for k_0, v_0 in vars_0.items():
        v_end = vars_end[k_0]
        # y = mx+y0 = y0 + (y_end - y_0) / 1 * n_cycles
        extrapolated_vars[k_0] = v_end + (v_end - v_0) / 1 * (n_cycles - 1)

    # print(vars_end, extrapolated_vars)

    return extrapolated_vars

import numpy as np
def update_input_params(param_input, extrapolated_vars, parameter_values):

    # dQn = (sol["Negative electrode capacity [A.h]"].entries[-1] * extrapolated_vars["eps_n"] /
    #        sol["X-averaged negative electrode active material volume fraction"].entries[-1])

    #param_input = {}
    # param_input["Initial concentration in negative electrode [mol.m-3]"] -= extrapolated_vars["LLI_sei"] / dQn * parameter_values["Maximum concentration in negative electrode [mol.m-3]"]

    eps_n = prev_params["Negative electrode active material volume fraction"]
    L = prev_params["Negative electrode thickness [m]"]
    A = prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"]
    cs_max = prev_params["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    Q0n = eps_n * L * cs_max * F * A / 3600
    dQn = (Q0n * extrapolated_vars["eps_n"] /
           prev_params["Negative electrode active material volume fraction"])

    param_input["Initial concentration in negative electrode [mol.m-3]"] = prev_params["Initial concentration in negative electrode [mol.m-3]"] - extrapolated_vars["LLI_sei"] / dQn * \
                                                                            parameter_values[
                                                                                "Maximum concentration in negative electrode [mol.m-3]"]

    param_input["Initial SEI thickness [m]"] = extrapolated_vars["L_sei"]

    # Load the extrapolated eps
    param_input["Negative electrode active material volume fraction"] =  extrapolated_vars["eps_n"]
    param_input["Positive electrode active material volume fraction"] =  extrapolated_vars["eps_p"]

    return param_input

COARSE_KEYS = ["LLI_sei", "L_sei", "eps_n", "eps_p"]

def y_dict_to_vec(y):
    return np.array([y[k] for k in COARSE_KEYS], dtype=float)

def y_vec_to_dict(v):
    return {k: float(v[i]) for i, k in enumerate(COARSE_KEYS)}

def eval_delta_y(y_vec, sim, param_input, parameter_values):

    # Convert vector → dict
    y_dict = y_vec_to_dict(y_vec)

    # Construct extrapolated_vars in your existing format
    extrapolated_vars = {
        "LLI_sei": y_dict["LLI_sei"],
        "L_sei": y_dict["L_sei"],
        "eps_n": y_dict["eps_n"],
        "eps_p": y_dict["eps_p"],
    }

    # Copy template solution and params
    param_input_sol = deepcopy(param_input)

    # Try direct injection into last_state
    param_input_sol = update_input_params(param_input_sol, extrapolated_vars, parameter_values)
    try:
        sol_fast = sim.solve(inputs=param_input_sol)
    except:
        sol_fast = sim.solve(inputs=param_input_sol)

    # Extract y0 and y_end
    y0_dict, yend_dict = extract_y0_yend(sol_fast.cycles[-1])

    # Convert end-state to vector
    y_end_vec = y_dict_to_vec(yend_dict)

    y_end_vec[:1] += y_vec[:1]

    # Δy = y_end − y
    return y_end_vec - y_vec, sol_fast, param_input_sol

def scipy_rk23_integrate(y0_vec, sol_0, sim,
                         param_input, parameter_values,
                         t_final, rtol=1e-2, atol=1e-2,
                         stop_condition_fn=None):
    """
    Integrates dy/dt = Δy(y) from t=0 to t=t_final using SciPy's RK23.
    One RHS evaluation = one cycle simulation.
    Returns list of (t, y_vec, full_cycle_solution)
    """

    # Wrapper converting SciPy call signature f(t,y) → dy/dt
    def rhs(t, y):
        dy, sol, param_input_sol = eval_delta_y(y, sim,
                                            rhs.param, parameter_values)
        # store last computed cycle
        rhs.last_sol = sol
        return dy

    rhs.last_sol = sol_0
    rhs.param = param_input

    solver = RK23(rhs, t0=0.0, y0=y0_vec, t_bound=t_final,
                  max_step=t_final, rtol=rtol, atol=atol)

    history = [(0.0, y0_vec.copy(), sol_0)]

    while solver.status == "running":
        solver.step()

        t = solver.t
        y = solver.y.copy()
        sol = rhs.last_sol          # full cycle solution used at this step

        history.append((t, y, sol))

        # Optional early stopping
        if stop_condition_fn is not None:
            if stop_condition_fn(t, y, sol):
                break

    return history

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
    parameter_values = pybamm.ParameterValues("Chen2020")
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
        # print(p, "updated to ", str(param_Ai[p]))

    #parameter_values.update({"Negative electrode partial molar volume [m3.mol-1]": param_Ai[]}, check_already_exists=False)
    #parameter_values.update({"Positive electrode partial molar volume [m3.mol-1]": 0.}, check_already_exists=False)

    parameter_values.set_initial_stoichiometries(1)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values)

    # spm = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited", "loss of active material": "stress-driven","x-average side reactions":"true"})
    spm = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"})
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

    param_input = {"Initial concentration in negative electrode [mol.m-3]": parameter_values[
                       "Initial concentration in negative electrode [mol.m-3]"],
                   "Initial SEI thickness [m]": parameter_values["Initial SEI thickness [m]"],
                   "Negative electrode active material volume fraction": parameter_values[
                       "Negative electrode active material volume fraction"],
                   "Positive electrode active material volume fraction": parameter_values[
                       "Positive electrode active material volume fraction"]}
    cs_ini = param_input["Initial concentration in negative electrode [mol.m-3]"]
    prev_params = deepcopy(parameter_values)


    parameter_values.update({"Initial concentration in negative electrode [mol.m-3]": "[input]",
                             "Initial SEI thickness [m]": "[input]",
                             "Negative electrode active material volume fraction": "[input]",
                             "Positive electrode active material volume fraction": "[input]",})


    sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=parameter_values)

    start_extrap_time = time.time()

    sol_0 = sim.solve(inputs=param_input)
    y0_vec, _ = extract_y0_yend(sol_0)
    y0_vec = y_dict_to_vec(y0_vec)

    history = scipy_rk23_integrate(
        y0_vec=y0_vec,
        sol_0=sol_0,  # template to modify
        sim=sim,
        param_input=param_input,
        parameter_values=parameter_values,
        t_final=float(tot_cycles),  # e.g. 1000
        rtol=1e-2,
        atol=1e-2,
        stop_condition_fn=None  # optional early-stop
    )
    extrap_time = time.time() - start_extrap_time

    print("Extrapolation time: ", extrap_time)

    all_Qs = [get_capacity_from_sol(s, prev_params, 0, esoh_solver) for (_, _, s) in history]
    cycles = [t for (t, _, _) in history]

    fig, ax = plt.subplots()
    ax.plot(Q_bs, label="Baseline")
    ax.scatter(cycles, all_Qs, label="ODE",
               c="red")
    #ax.scatter(list(range(len(extrapolation_results["Q"]))), extrapolation_results["Q"], label="Extrapolated N=1",
    #           c="green")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Discharge capacity [A.h]")
    plt.legend()
    plt.show()

