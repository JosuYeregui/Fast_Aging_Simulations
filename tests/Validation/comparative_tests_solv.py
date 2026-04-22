import pybamm
from matplotlib import pyplot as plt
import os
from copy import deepcopy
import time

from main import FastAging_Simulation
from utils.param_to_inputvars import load_params
from utils.cycler import CCStep, CCCVStep, Cycler, DynamicProfile, Rest
from utils.characterize import *
from utils.pybamm_submodels import *

import copy

from scipy.integrate import RK23
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import scienceplots
plt.style.use(['science'])

from utils.pybamm_submodels import update_LAM_params, StressBasedLAM_FP, Plating_FP


def get_capacity_from_sol(solution, param, idx, esoh_solver):
    x_0, y_0, x_100, y_100 = extract_stoich_from_sol(solution, idx, esoh_solver)
    eps_n = solution["X-averaged negative electrode active material volume fraction"].entries[idx]
    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    #return solution["X-averaged negative dead lithium thickness [m]"].entries[idx]
    #return solution["X-averaged negative dead lithium concentration [mol.m-3]"].entries[idx]
    return eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600, [x_0, y_0, x_100, y_100]

def extract_stoich_from_sol(solution, idx, esoh_solver):
    eps_n = solution["X-averaged negative electrode active material volume fraction"].entries[idx]
    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    Qn = eps_n * L * cs_max * F * A / 3600.

    eps_p = solution["X-averaged positive electrode active material volume fraction"].entries[idx]
    L = param["Positive electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in positive electrode [mol.m-3]"]
    F = 96485.33

    Qp = eps_p * L * cs_max * F * A / 3600.

    #Qn = solution["Negative electrode capacity [A.h]"].entries[idx]
    #Qp = solution["Positive electrode capacity [A.h]"].entries[idx]
    Qp_0 = solution["X-averaged positive particle concentration"].entries[-1][idx] * \
           solution["Positive electrode capacity [A.h]"].entries[idx]
    Qn_0 = solution["X-averaged negative particle concentration"].entries[-1][idx] * \
           solution["Negative electrode capacity [A.h]"].entries[idx]
    QLi = (Qp_0 + Qn_0)

    inputs = {"V_min": 2.5, "V_max": 4.2,
              "Q_n": Qn,
              "Q_p": Qp,
              "Q_Li": QLi}

    esoh_sol = esoh_solver.solve(inputs)

    x_0 = esoh_sol["x_0"]
    y_0 = esoh_sol["y_0"]
    x_100 = esoh_sol["x_100"]
    y_100 = esoh_sol["y_100"]

    return x_0, y_0, x_100, y_100

def extract_stoich_from_param(param, extrap_vars):
    model = pybamm.BaseModel()
    x_100 = pybamm.Variable("x_100")
    y_100 = pybamm.Variable("y_100")
    x_0 = pybamm.Variable("x_0")
    y_0 = pybamm.Variable("y_0")

    eps_n = extrap_vars["eps_n"]
    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    Qn = eps_n * L * cs_max * F * A / 3600.

    eps_p = extrap_vars["eps_p"]
    L = param["Positive electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in positive electrode [mol.m-3]"]
    F = 96485.33

    Qp = eps_p * L * cs_max * F * A / 3600.

    n_Li = extrap_vars["n_Li"]

    U_p = param["Positive electrode OCP [V]"]
    U_n = param["Negative electrode OCP [V]"]

    Vmax = 4.2
    Vmin = 2.5

    model.algebraic = {x_100: U_p(y_100) - U_n(x_100) - Vmax,
                       y_100: n_Li - 3600/F * (x_100 * Qn + y_100 * Qp),
                       x_0: Qn*(x_100 - x_0) - Qp*(y_0 - y_100),
                       y_0: U_p(y_0) - U_n(x_0) - Vmin}

    model.variables = {"x_100": x_100, "y_100": y_100, "x_0": x_0, "y_0": y_0}
    model.initial_conditions = {x_100: 0.5, y_100: 0.5, x_0: 0.5, y_0: 0.5}
    sim = pybamm.Simulation(model, parameter_values=param)
    sol = sim.solve([0])

    x_100 = sol["x_100"].data[0]
    y_100 = sol["y_100"].data[0]
    x_0 = sol["x_0"].data[0]
    y_0 = sol["y_0"].data[0]

    return x_0, y_0, x_100, y_100

def extract_y0_yend(sol):
    res = extract_y_from_solution(sol)
    return {k: v[0] for k, v in res.items()}, {k: v[-1] for k, v in res.items()}

def extract_y_from_solution(sol):
    eps_n = sol["X-averaged negative electrode active material volume fraction"].entries
    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    Qn = eps_n * L * cs_max * F * A / 3600.

    eps_p = sol["X-averaged positive electrode active material volume fraction"].entries
    L = param["Positive electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in positive electrode [mol.m-3]"]
    F = 96485.33

    Qp = eps_p * L * cs_max * F * A / 3600.

    L_sei = sol["X-averaged negative SEI thickness [m]"].entries
    n_Li = (sol["Average negative particle concentration [mol.m-3]"].entries / param[
        "Maximum concentration in negative electrode [mol.m-3]"] * Qn +
            sol["Average positive particle concentration [mol.m-3]"].entries / param[
                "Maximum concentration in positive electrode [mol.m-3]"] * Qp) * 3600 / 96485.33
    # c_pl = sol["X-averaged negative dead lithium concentration [mol.m-3]"].entries + sol["X-averaged positive dead lithium concentration [mol.m-3]"].entries
    L_pl = sol["X-averaged negative dead lithium thickness [m]"].entries
    eps_n = sol["X-averaged negative electrode active material volume fraction"].entries
    eps_p = sol["X-averaged positive electrode active material volume fraction"].entries

    return {"n_Li": n_Li, "L_sei": L_sei, "L_pl": L_pl,
            "eps_n": eps_n, "eps_p": eps_p}


import numpy as np
def update_input_params(param_input, extrapolated_vars, parameter_values):

    x_0, y_0, x_100, y_100 = extract_stoich_from_param(prev_params, extrapolated_vars)

    param_input["Initial concentration in negative electrode [mol.m-3]"] = x_100 * parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
    param_input["Initial concentration in positive electrode [mol.m-3]"] = y_100 * parameter_values["Maximum concentration in positive electrode [mol.m-3]"]

    param_input["Initial SEI thickness [m]"] = extrapolated_vars["L_sei"]

    a_n = extrapolated_vars["eps_n"] * 3 / parameter_values["Negative particle radius [m]"]
    V_Li = parameter_values["Lithium metal partial molar volume [m3.mol-1]"]
    param_input["Initial plated lithium concentration [mol.m-3]"] = extrapolated_vars["L_pl"] * a_n / V_Li

    # Load the extrapolated eps
    param_input["Negative electrode active material volume fraction"] =  extrapolated_vars["eps_n"]
    param_input["Positive electrode active material volume fraction"] =  extrapolated_vars["eps_p"]

    return param_input

COARSE_KEYS = ["n_Li", "L_sei", "L_pl", "eps_n", "eps_p"]

def y_dict_to_vec(y):
    return np.array([y[k] for k in COARSE_KEYS], dtype=float)

def y_vec_to_dict(v):
    return {k: float(v[i]) for i, k in enumerate(COARSE_KEYS)}

def eval_delta_y(y_vec, sim, param_input, parameter_values):

    y_dict = y_vec_to_dict(y_vec)

    extrapolated_vars = {
        "n_Li": y_dict["n_Li"],
        "L_sei": y_dict["L_sei"],
        "L_pl": y_dict["L_pl"],
        "eps_n": y_dict["eps_n"],
        "eps_p": y_dict["eps_p"],
    }

    param_input_sol = deepcopy(parameter_values)

    param_input_sol = update_input_params(param_input_sol, extrapolated_vars, parameter_values)
    sol_fast = sim.solve(inputs=param_input_sol)

    y0_dict, yend_dict = extract_y0_yend(sol_fast.cycles[-1])
    y_end_vec = y_dict_to_vec(yend_dict)

    #y_end_vec[0] += y_vec[0] # LLI is given as delta from sol

    return y_end_vec - y_vec, sol_fast, param_input_sol

def scipy_rk23_integrate(y0_vec, sol_0, sim,
                         param_input, parameter_values,
                         t_final, max_step, rtol=1e-2, atol=1e-2,
                         stop_condition_fn=None):

    # Wrapper converting SciPy call signature f(t,y) → dy/dt
    def rhs(t, y):
        dy, sol, param_input_sol = eval_delta_y(y, sim,
                                            rhs.param, parameter_values)
        # store last computed cycle
        rhs.last_sol = sol
        return dy

    #rhs.last_sol = sol_0
    rhs.param = param_input

    solver = RK23(rhs, t0=0.0, y0=y0_vec, t_bound=t_final,
                  max_step=max_step, rtol=rtol, atol=atol)

    history = []#(0.0, y0_vec.copy(), sol_0)]
    idx = 1

    while solver.status == "running":
        solver.step()

        t = solver.t
        y = solver.y.copy()
        sol = rhs.last_sol          # full cycle solution used at this step

        history.append((t, y, sol))
        idx += 1

        if y[3] < 0.1 or y[4] < 0.1:
            break

        if stop_condition_fn is not None:
            if stop_condition_fn(t, y, sol):
                break

    return history

def extrapolate_solution(solution, n_cycles):

    vars_0, vars_end = extract_y0_yend(solution)

    extrapolated_vars = {}
    for k_0, v_0 in vars_0.items():
        v_end = vars_end[k_0]
        # y = mx+y0 = y0 + (y_end - y_0) / 1 * n_cycles
        extrapolated_vars[k_0] = v_end + (v_end - v_0) / 1 * (n_cycles - 1)

    # print(vars_end, extrapolated_vars)

    return extrapolated_vars


if __name__ == "__main__":


    tot_cycles = 100 # Total cycles to simulate
    initial_soc = 1.
    save_fig = False


    # -------------- Define model ---------------------

    param_chen = pybamm.ParameterValues("Chen2020")
    param = pybamm.ParameterValues("OKane2022")
    param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
    param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
    param["Electrolyte diffusivity [m2.s-1]"] = param["Electrolyte diffusivity [m2.s-1]"](1., 298.15)

    param["EC diffusivity [m2.s-1]"] /= 150
    param["Positive electrode LAM constant proportional term [s-1]"] /= 2000.
    param["Negative electrode LAM constant proportional term [s-1]"] *= 20.
    param["Lithium plating kinetic rate constant [m.s-1]"] /= 100.

    param["Upper voltage cut-off [V]"] = 4.2
    param = update_LAM_params(param)
    param.set_initial_state(initial_soc)
    prev_params = deepcopy(param)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    model = pybamm.lithium_ion.SPMe(
        options={"SEI": "ec reaction limited", "x-average side reactions": "false"},
        build=False)

    model.submodels["Negative active material"] = StressBasedLAM(
        model.param, "Negative"
    )

    model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.submodels["Plating"] = Plating_FP(model.param, "negative", x_average=True, options={"lithium plating": "irreversible"})

    model.submodels["Positive active material"] = StressBasedLAM(
        model.param, "Positive"
    )

    model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.build_model()


    # -------------- Baseline ---------------------


    exp = ("Discharge at 1C until 3 V", "Rest for 1 hour", "Charge at 0.3C until 4.2 V","Hold at 4.2V until C/50")

    experiment = pybamm.Experiment([exp] * tot_cycles)
    start_time = time.time()
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
    base_sol = sim.solve(initial_soc=initial_soc)
    print("Baseline time: ", time.time() - start_time)

    n_Li0 = (base_sol["Average negative particle concentration [mol.m-3]"].entries / param["Maximum concentration in negative electrode [mol.m-3]"] * base_sol["Negative electrode capacity [A.h]"].entries + base_sol["Average positive particle concentration [mol.m-3]"].entries / param["Maximum concentration in positive electrode [mol.m-3]"] * base_sol[
               "Positive electrode capacity [A.h]"].entries) * 3600 / 96485.33
    n_Li0 = n_Li0[0]


    # -------------- Sulzer ---------------------


    extrapolation_results = {"cycle": [], "Q": [], "eps_n": [], "eps_p": [], "L_sei": [], "LLI_sei": [], "nLi":[]}
    starting_sol = None

    param_input = {"Initial concentration in negative electrode [mol.m-3]": param[
                       "Initial concentration in negative electrode [mol.m-3]"],
                   "Initial concentration in positive electrode [mol.m-3]": param[
                       "Initial concentration in positive electrode [mol.m-3]"],
                   "Initial SEI thickness [m]": param["Initial SEI thickness [m]"],
                   "Initial plated lithium concentration [mol.m-3]": param["Initial plated lithium concentration [mol.m-3]"]*0,
                   "Negative electrode active material volume fraction": param[
                       "Negative electrode active material volume fraction"],
                   "Positive electrode active material volume fraction": param[
                       "Positive electrode active material volume fraction"],}
    cs_ini = param_input["Initial concentration in negative electrode [mol.m-3]"]

    param.update({"Initial concentration in negative electrode [mol.m-3]": "[input]",
                  "Initial concentration in positive electrode [mol.m-3]": "[input]",
                  "Initial SEI thickness [m]": "[input]",
                  "Initial plated lithium concentration [mol.m-3]": "[input]",
                  "Negative electrode active material volume fraction": "[input]",
                  "Positive electrode active material volume fraction": "[input]",})

    experiment = pybamm.Experiment([exp])
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)

    y0_vec = y_dict_to_vec({"n_Li": n_Li0,
                            "L_sei": prev_params["Initial SEI thickness [m]"],
                            "L_pl": 0.,
                            "eps_n": prev_params["Negative electrode active material volume fraction"],
                            "eps_p": prev_params["Positive electrode active material volume fraction"],})

    start_extrap_time = time.time()
    results_sulzer = scipy_rk23_integrate(
        y0_vec=y0_vec,
        sol_0=None,  # template to modify
        sim=sim,
        param_input=param_input,
        parameter_values=param,
        t_final=float(tot_cycles),  # e.g. 1000
        max_step=tot_cycles,
        rtol=1e-3,
        atol=1e-3,
        stop_condition_fn=None  # optional early-stop
    )
    extrap_time = time.time() - start_extrap_time
    print("Sulzer time: ", extrap_time)


    # -------------- Proposed solution ---------------------


    input_vars = load_params("../../input/M50_okane_modified_plating.yaml")
    input_vars["D_ec"] /= 150
    # input_vars["k_sei"] /= 1000
    input_vars["Beta_n"] *= 20
    input_vars["Beta_p"] /= 2000
    input_vars["k_plating"] /= 100
    input_vars["D_n"] *= 1
    # input_vars["U_sei"] = 0.4

    Q0 = input_vars["Q"]
    Q = copy.deepcopy(Q0)
    R = input_vars["R_int"]
    OCV_fn = get_OVC_func(input_vars)

    steps = [
        CCStep(I=-5 * 1, V_end=3),
        Rest(t=1.),
        # CCStep(I=5, V_end=4.2),
        CCCVStep(I=5 * 0.3, V_end=4.2, Cutoff_I=5 / 50, Vlim=[2.5, 4.2]),
    ]

    cycl = Cycler(
        steps=steps,
        SOC0=initial_soc,
        R=R,
        Q=Q,
        OCV_fn=OCV_fn
    )

    soc_ave = cycl.get_soc_ave()
    sol_n = soc_ave * (input_vars["x_100"] - input_vars["x_0"]) + input_vars["x_0"]
    input_vars["cini_n_0"] = sol_n * input_vars["csmax_n"]
    input_vars["cini_n"] = sol_n * input_vars["csmax_n"]
    sol_p = soc_ave * (input_vars["y_100"] - input_vars["y_0"]) + input_vars["y_0"]
    input_vars["cini_p_0"] = sol_p * input_vars["csmax_p"]
    input_vars["cini_p"] = sol_p * input_vars["csmax_p"]

    y0_vec = [input_vars["L_0"], input_vars["eps_0_n"], input_vars["eps_0_p"], 0.]  # L, eps_n, eps_p, Plating

    input_vars["T"] += 0.

    start_ode_time = time.time()
    history_ode = FastAging_Simulation(
        y0_vec=y0_vec,
        input_vars=input_vars,
        tot_cycles=tot_cycles,
        cycler=cycl,
        rtol=[1e-5, 1e-3, 1e-3, 1e-5],
        atol=[1e-9, 1e-4, 1e-4, 1e-9],
        stop_condition_fn=None,  # optional early-stop
        max_step=None
    )
    ode_time = time.time() - start_ode_time
    print("ODE + integration time: ", ode_time)


    # -------------- Process results ---------------------


    # t_ode = [h[0] for h in history]
    t_ode = history_ode["t [s]"] / 3600.
    cycles_ode = history_ode["n_cycles"]
    Lsei_ode = history_ode["y"][:, 0]  * 1e9
    eps_n_ode = history_ode["y"][:, 1]
    LAMn_ode = (1 - eps_n_ode / prev_params["Negative electrode active material volume fraction"]) * 100.
    eps_p_ode = history_ode["y"][:, 2]
    LAMp_ode = (1 - eps_p_ode / prev_params["Positive electrode active material volume fraction"]) * 100.
    Lpl_ode = history_ode["y"][:, 3] * 1e9
    t_final_evol = history_ode["times"]
    Q_ode = history_ode["Q"][:]
    stoich_ode = np.asarray(history_ode["stoichiometry"])

    Q_sulzer = []
    stoich_sulzer = []
    for (_, _, s) in results_sulzer:
        q, st = get_capacity_from_sol(s, prev_params, 0, esoh_solver)
        Q_sulzer.append(q)
        stoich_sulzer.append(st)
    Q_sulzer = np.asarray(Q_sulzer)
    stoich_sulzer = np.asarray(stoich_sulzer)
    cycles_sulzer = [t for (t, _, _) in results_sulzer]
    Lpl_sulzer = np.asarray([s["X-averaged negative dead lithium thickness [m]"].entries[0] for (_, _, s) in results_sulzer]) * 1e9
    eps_n_sulzer = np.asarray([s["X-averaged negative electrode active material volume fraction"].entries[0] for (_, _, s) in results_sulzer])
    LAMn_sulzer = (1 - eps_n_sulzer / prev_params["Negative electrode active material volume fraction"]) * 100.
    eps_p_sulzer = np.asarray([s["X-averaged positive electrode active material volume fraction"].entries[0] for (_, _, s) in results_sulzer])
    LAMp_sulzer = (1 - eps_p_sulzer / prev_params["Positive electrode active material volume fraction"]) * 100.
    Lsei_sulzer = np.asarray([s["X-averaged negative SEI thickness [m]"].entries[0] for (_, _, s) in results_sulzer]) * 1e9


    cycles_bs = np.arange(len(base_sol.cycles))
    Q_bs = []
    stoich_bs = []
    for s in base_sol.cycles:
        q, st = get_capacity_from_sol(s, prev_params, 0, esoh_solver)
        Q_bs.append(q)
        stoich_bs.append(st)
    Q_bs = np.asarray(Q_bs)
    stoich_bs = np.asarray(stoich_bs)
    Lpl_bs = np.asarray([s["X-averaged negative dead lithium thickness [m]"].entries[0] for s in base_sol.cycles]) * 1e9
    epsn_bs = np.asarray([s["X-averaged negative electrode active material volume fraction"].entries[0] for s in base_sol.cycles])
    LAMn_bs = (1 - epsn_bs / prev_params["Negative electrode active material volume fraction"]) * 100.
    epsp_bs = np.asarray([s["X-averaged positive electrode active material volume fraction"].entries[0] for s in base_sol.cycles])
    LAMp_bs = (1 - epsp_bs / prev_params["Positive electrode active material volume fraction"]) * 100.
    Lsei_bs = np.asarray([s["X-averaged negative SEI thickness [m]"].entries[0] for s in base_sol.cycles]) * 1e9
    por_n_bs = np.asarray([s["X-averaged negative electrode porosity"].entries[0] for s in base_sol.cycles])
    por_p_bs = np.asarray([s["X-averaged positive electrode porosity"].entries[0] for s in base_sol.cycles])
    dead_lithium_n = np.asarray([s["X-averaged negative dead lithium concentration [mol.m-3]"].entries[0] for s in base_sol.cycles])
    dead_lithium_p = np.asarray([s["X-averaged positive dead lithium concentration [mol.m-3]"].entries[0] for s in base_sol.cycles])
    #print(dead_lithium_n, dead_lithium_p)
    #print(por_p, por_n)


    variables_to_interpolate_ode = {
        "Lsei": Lsei_ode,
        "eps_n": eps_n_ode,
        "LAMn": LAMn_ode,
        "eps_p": eps_p_ode,
        "LAMp": LAMp_ode,
        "Lpl": Lpl_ode,
        "Q": Q_ode
    }

    interpolated_results_ode = {}
    target_cycle = tot_cycles

    for name, data in variables_to_interpolate_ode.items():
        interpolated_results_ode[name] = np.interp(target_cycle, cycles_ode, data)

    variables_to_interpolate_sulzer = {
        "Lsei": Lsei_sulzer,
        "eps_n": eps_n_sulzer,
        "LAMn": LAMn_sulzer,
        "eps_p": eps_p_sulzer,
        "LAMp": LAMp_sulzer,
        "Lpl": Lpl_sulzer,
        "Q": Q_sulzer
    }

    interpolated_results_sulzer = {}
    for name, data in variables_to_interpolate_sulzer.items():
        interpolated_results_sulzer[name] = np.interp(target_cycle, cycles_sulzer, data)

    stoich_ode_interp = np.array([np.interp(target_cycle, cycles_ode, stoich_ode[:, col]) for col in range(stoich_ode.shape[1])])
    stoich_sulzer_interp = np.array([np.interp(target_cycle, cycles_sulzer, stoich_sulzer[:, col]) for col in range(stoich_sulzer.shape[1])])

    results_bs = {
        "Lsei": Lsei_bs[tot_cycles-1],
        "eps_n": epsn_bs[tot_cycles-1],
        "LAMn": LAMn_bs[tot_cycles-1],
        "eps_p": epsp_bs[tot_cycles-1],
        "LAMp": LAMp_bs[tot_cycles-1],
        "Lpl": Lpl_bs[tot_cycles-1],
        "Q": Q_bs[tot_cycles-1]
    }

    for i, (ode, sul, bs) in enumerate(zip(interpolated_results_ode.items(), interpolated_results_sulzer.items(), results_bs.items())):
        name = ode[0]
        error_sul = abs(sul[1] - bs[1]) / abs(bs[1]) * 100 if bs[1] != 0 else float('inf')
        error_ode = abs(ode[1] - bs[1]) / abs(bs[1]) * 100 if bs[1] != 0 else float('inf')
        print(f"{name}: Error Sulzer={error_sul:.2f}%, Proposed={error_ode:.2f}%")
        variables_to_interpolate_ode[name] = np.append(variables_to_interpolate_ode[name], ode[1])
        variables_to_interpolate_sulzer[name] = np.append(variables_to_interpolate_sulzer[name], sul[1])
        if i == 0:
            cycles_ode = np.append(cycles_ode, tot_cycles)
            cycles_sulzer = np.append(cycles_sulzer, tot_cycles)

    Lsei_ode, eps_n_ode, LAMn_ode, eps_p_ode, LAMp_ode, Lpl_ode, Q_ode = (
        variables_to_interpolate_ode[k] for k in ["Lsei", "eps_n", "LAMn", "eps_p", "LAMp", "Lpl", "Q"]
    )
    Lsei_sulzer, eps_n_sulzer, LAMn_sulzer, eps_p_sulzer, LAMp_sulzer, Lpl_sulzer, Q_sulzer = (
        variables_to_interpolate_sulzer[k] for k in ["Lsei", "eps_n", "LAMn", "eps_p", "LAMp", "Lpl", "Q"]
    )
    stoich_ode = np.vstack([stoich_ode, stoich_ode_interp])
    stoich_sulzer = np.vstack([stoich_sulzer, stoich_sulzer_interp])

    fig, ax = plt.subplots(figsize=(2.1, 2.5))
    ax.plot(cycles_bs, Q_bs, label="Baseline", color="black")
    ax.scatter(cycles_sulzer, Q_sulzer, label="[X]", color="black", marker="x")
    ax.scatter(cycles_ode, Q_ode, label="Proposed", color="black")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$Q [Ah]$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, tot_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('Results/Q.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(2.1, 2.5))
    ax.plot(cycles_bs, LAMn_bs, label="Baseline", color="blue")
    ax.scatter(cycles_sulzer, LAMn_sulzer, label="[X]", color="blue", marker="x")
    ax.scatter(cycles_ode, LAMn_ode, label="Proposed", color="blue")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$LAM_n [\%]$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, tot_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('Results/LAMn.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(2.1, 2.5))
    ax.plot(cycles_bs, LAMp_bs, label="Baseline", color="orange")
    ax.scatter(cycles_sulzer, LAMp_sulzer, label="[X]", color="orange", marker="x")
    ax.scatter(cycles_ode, LAMp_ode, label="Proposed", color="orange")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$LAM_p [\%]$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, tot_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('Results/LAMp.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(2.1, 2.5))
    ax.plot(cycles_bs, Lpl_bs, label="Baseline", color="green")
    ax.scatter(cycles_sulzer, Lpl_sulzer, label="[X]", color="green", marker="x")
    ax.scatter(cycles_ode, Lpl_ode, label="Proposed", color="green")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$L_{Plating} [nm]$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, tot_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('Results/Lplating.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(2.1, 2.5))
    ax.plot(cycles_bs, Lsei_bs, label="Baseline", color="lightgreen")
    ax.scatter(cycles_sulzer, Lsei_sulzer, label="[X]", color="lightgreen", marker="x")
    ax.scatter(cycles_ode, Lsei_ode, label="Proposed", color="lightgreen")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$L_{SEI} [nm]$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, tot_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('Results/Lsei.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots(figsize=(2.1, 2.5))
    ax.plot(cycles_bs, stoich_bs[:, 1], label="$y_0$", color="red")
    ax.scatter(cycles_sulzer, stoich_sulzer[:, 1], color="red", marker="x")
    ax.scatter(cycles_ode, stoich_ode[:, 2], color="red")

    ax.plot(cycles_bs, stoich_bs[:, 2], label="$x_{100}$", color="cyan")
    ax.scatter(cycles_sulzer, stoich_sulzer[:, 2], color="cyan", marker="x")
    ax.scatter(cycles_ode, stoich_ode[:, 1], color="cyan")

    ax.set_xlabel("Cycles")
    ax.set_ylabel(r"$\theta$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, tot_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('Results/stoich.pdf', dpi=300)
    plt.show()