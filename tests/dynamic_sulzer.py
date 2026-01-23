import pybamm
from matplotlib import pyplot as plt
import os
from copy import deepcopy
import time

from scipy.integrate import RK23
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utils.pybamm_submodels import update_LAM_params, StressBasedLAM_FP

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

    y_dict = y_vec_to_dict(y_vec)

    extrapolated_vars = {
        "LLI_sei": y_dict["LLI_sei"],
        "L_sei": y_dict["L_sei"],
        "eps_n": y_dict["eps_n"],
        "eps_p": y_dict["eps_p"],
    }

    param_input_sol = deepcopy(param_input)

    param_input_sol = update_input_params(param_input_sol, extrapolated_vars, parameter_values)
    sol_fast = sim.solve(inputs=param_input_sol)

    y0_dict, yend_dict = extract_y0_yend(sol_fast.cycles[-1])
    y_end_vec = y_dict_to_vec(yend_dict)

    y_end_vec[0] += y_vec[0] # LLI is given as delta from sol

    return y_end_vec - y_vec, sol_fast, param_input_sol

def scipy_rk23_integrate(y0_vec, sol_0, sim,
                         param_input, parameter_values,
                         t_final, rtol=1e-2, atol=1e-2,
                         stop_condition_fn=None):

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
    idx = 1

    while solver.status == "running":
        startTime = time.time()
        solver.step()
        endTime = time.time() - startTime

        t = solver.t
        y = solver.y.copy()
        sol = rhs.last_sol          # full cycle solution used at this step

        history.append((t, y, sol))
        print("Completed step: ", idx, " in ", endTime, " seconds")
        idx += 1

        if y[2] < 0.1 or y[3] < 0.1:
            break

        if stop_condition_fn is not None:
            if stop_condition_fn(t, y, sol):
                break

    return history

# sol_baseline.cycles[-1].steps[-1].y_slices
if __name__ == "__main__":

    tot_cycles = 1000 # Total cycles to simulate
    n_cycles = 25 # each how many cycles to extrapolate

    param_chen = pybamm.ParameterValues("Chen2020")
    param = pybamm.ParameterValues("OKane2022")
    param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
    param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
    # param["Ratio of lithium moles to SEI moles"] = 2.

    param["EC diffusivity [m2.s-1]"] /= 10
    param["Positive electrode LAM constant proportional term [s-1]"] /= 100.
    param["Negative electrode LAM constant proportional term [s-1]"] *= 200.
    param["Upper voltage cut-off [V]"] = 4.2

    param = update_LAM_params(param)

    param.set_initial_stoichiometries(1)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    # spm = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited", "loss of active material": "stress-driven","x-average side reactions":"true"})

    spm = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"},
                                    build=False)

    # model.submodels["Negative active material"] = StressBasedLAM(
    #     model.param, "Negative"
    # )

    spm.submodels["Negative active material"] = StressBasedLAM_FP(spm.param, "Negative", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")

    # model.submodels["Positive active material"] = StressBasedLAM(
    #     model.param, "Positive"
    # )

    spm.submodels["Positive active material"] = StressBasedLAM_FP(spm.param, "Positive", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")

    spm.build_model()


    extrapolation_results = {"cycle": [], "Q": [], "eps_n": [], "eps_p": [], "L_sei": [], "LLI_sei": [], "nLi":[]}
    starting_sol = None

    param_input = {"Initial concentration in negative electrode [mol.m-3]": param[
                       "Initial concentration in negative electrode [mol.m-3]"],
                   "Initial SEI thickness [m]": param["Initial SEI thickness [m]"],
                   "Negative electrode active material volume fraction": param[
                       "Negative electrode active material volume fraction"],
                   "Positive electrode active material volume fraction": param[
                       "Positive electrode active material volume fraction"]}
    cs_ini = param_input["Initial concentration in negative electrode [mol.m-3]"]
    prev_params = deepcopy(param)

    cycle = np.loadtxt("../Dynamic_Profiles/WLTC.txt", skiprows=1)
    cycle[:, 0] -= 1
    cycle[:, 1] *= -1
    # drive_step = pybamm.step.BaseStep(cycle, termination="Voltage < 2.5V")

    drive_step = pybamm.step.current(cycle, termination="<2.5V", duration=1800000)

    experiment = pybamm.Experiment(["Discharge at 1 A until 3 V"])


    param.update({"Initial concentration in negative electrode [mol.m-3]": "[input]",
                  "Initial SEI thickness [m]": "[input]",
                  "Negative electrode active material volume fraction": "[input]",
                  "Positive electrode active material volume fraction": "[input]",})


    sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=param)

    start_extrap_time = time.time()

    sol_0 = sim.solve(inputs=param_input)
    y0_vec, _ = extract_y0_yend(sol_0)
    y0_vec = y_dict_to_vec(y0_vec)

    experiment = drive_step
    sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=param)

    history = scipy_rk23_integrate(
        y0_vec=y0_vec,
        sol_0=sol_0,  # template to modify
        sim=sim,
        param_input=param_input,
        parameter_values=param,
        t_final=float(tot_cycles),  # e.g. 1000
        rtol=1e-2,
        atol=1e-2,
        stop_condition_fn=None  # optional early-stop
    )
    extrap_time = time.time() - start_extrap_time

    import pickle

    # After running history = scipy_rk23_integrate(...)
    filename = "simulation_history_luzea.pkl"

    with open(filename, "wb") as f:
        pickle.dump(history, f)

    print("ODE time: ", extrap_time)

    all_Qs = [get_capacity_from_sol(s, prev_params, 0, esoh_solver) for (_, _, s) in history]
    cycles = [t for (t, _, _) in history]

    fig, ax = plt.subplots()
    ax.scatter(cycles, all_Qs, label="ODE",
               c="red")
    #ax.scatter(list(range(len(extrapolation_results["Q"]))), extrapolation_results["Q"], label="Extrapolated N=1",
    #           c="green")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Discharge capacity [A.h]")
    plt.legend()
    plt.show()

