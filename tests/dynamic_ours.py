import time
from matplotlib import pyplot as plt
import copy
import pickle

from main import scipy_rk23_integrate
from utils.param_to_inputvars import load_params
from utils.cycler import CCStep, CCCVStep, Cycler, DynamicProfile
from utils.characterize import *
from utils.pybamm_submodels import *

def extract_stoich_from_sol(solution, idx, esoh_solver):

    Qn = solution["Negative electrode capacity [A.h]"].entries[idx]
    Qp = solution["Positive electrode capacity [A.h]"].entries[idx]
    Qp_0 = solution["Average positive particle concentration"].entries[idx] * \
           solution["Positive electrode capacity [A.h]"].entries[idx]
    Qn_0 = solution["Average negative particle concentration"].entries[idx] * \
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

def get_capacity_from_sol(solution, p, idx, esoh_solver):
    x_0, _, x_100, _ = extract_stoich_from_sol(solution, idx, esoh_solver)
    eps_n = solution["X-averaged negative electrode active material volume fraction"].entries[idx]
    L = p["Negative electrode thickness [m]"]
    A = p["Electrode height [m]"] * p["Electrode width [m]"]
    cs_max = p["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    return eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600

def extract_y0_yend(sol):
    res = extract_y_from_solution(sol)
    return {k: v[0] for k, v in res.items()}, {k: v[-1] for k, v in res.items()}

def extract_y_from_solution(sol):

    LLI_sei = sol["Loss of capacity to negative SEI [A.h]"].entries
    L_sei = sol["X-averaged negative SEI thickness [m]"].entries
    eps_n = sol["X-averaged negative electrode active material volume fraction"].entries
    eps_p = sol["X-averaged positive electrode active material volume fraction"].entries

    return {"LLI_sei": LLI_sei, "L_sei": L_sei, "eps_n": eps_n, "eps_p": eps_p}


if __name__ == "__main__":

    tot_cycles = 10000
    initial_soc = 1.

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

    cycle = np.loadtxt("../Dynamic_Profiles/WLTC.txt", skiprows=1)
    cycle[:, 0] -= 1
    cycle[:, 1] *= -1

    model = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"}, build=False)

    model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.build_model()

    input_vars = load_params("../input/M50_okane_modified.yaml")
    input_vars["D_ec"] /= 10
    input_vars["Beta_n"] *= 200
    input_vars["Beta_p"] /= 100

    Q0 = input_vars["Q"]
    Q = copy.deepcopy(Q0)
    R = input_vars["R_int"]
    OCV_fn = get_OVC_func(input_vars)

    y0_vec = [input_vars["L_0"], input_vars["eps_0_n"], input_vars["eps_0_p"]]

    steps = [
        CCStep(I=1.5, V_end=3.5),
        CCCVStep(I=1, V_end=4.2, Cutoff_I=0.2, Vlim=[2.5, 4.2]),
        CCStep(I=-1, V_end=3.)
    ]
    # steps = [
    #     CCCVStep(I=1.5, V_end=4., Cutoff_I=0.2, Vlim=[2.5, 4.2]),
    #     DynamicProfile(V_lim=[2.5, 5.], cycle=True, dir="./Dynamic_Profiles/WLTC.txt")
    # ]
    steps = [
        DynamicProfile(V_lim=[2.5, 5.], cycle=True, dir="../Dynamic_Profiles/WLTC.txt"),
        CCStep(I=1, V_end=4.2)
    ]
    from copy import deepcopy as copy
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
    sol_p = soc_ave * (input_vars["y_100"] - input_vars["y_0"]) + input_vars["y_0"]
    input_vars["cini_p_0"] = sol_p * input_vars["csmax_p"]

    start_ode_time = time.time()
    history = scipy_rk23_integrate(
        y0_vec=y0_vec,
        input_vars=input_vars,
        tot_cycles=tot_cycles,
        cycler=cycl,
        rtol=[1e-3, 1e-3, 1e-3],
        atol=[1e-9 ,1e-4, 1e-4],
        stop_condition_fn=None,  # optional early-stop
        max_step=None
    )
    ode_time = time.time() - start_ode_time
    print("ODE + integration time: ", ode_time)

    #t_ode = [h[0] for h in history]
    t_ode = history["n_cycles"]
    L_ode = history["y"][:, 0]
    eps_n_ode = history["y"][:, 1]
    eps_p_ode = history["y"][:, 2]
    t_final_evol = history["times"]
    LLI = history["LLI"]
    Q_ode = history["Q"]

    with open("simulation_history_luzea_wCharge.pkl", "rb") as f:
        sulzer = pickle.load(f)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    all_Qs = [get_capacity_from_sol(s, param, 0, esoh_solver) for (_, _, s) in sulzer]
    LAMp = [y[3] for (_, y, _) in sulzer]
    LAMn = [y[2] for (_, y, _) in sulzer]
    Lsei = [y[1] for (_, y, _) in sulzer]
    cycles = [t for (t, _, _) in sulzer]

    fig, ax = plt.subplots()
    ax.plot(cycles, Lsei, label="Baseline")
    ax.scatter(np.asarray(t_ode), L_ode, label="ODE",
               c="red")
    #ax.scatter(np.asarray(t_ode)*cycl.total_time, L_ode, label="ODE",
    #           c="red")
    ax.set_xlabel("cycles")
    ax.set_ylabel("L SEI [m]")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(cycles, LAMn, label="\epsilon_n Baseline", color="blue")
    ax.plot(np.asarray(t_ode), eps_n_ode, label="\epsilon_n ODE",
               c="blue")
    ax.scatter(cycles, LAMp, label="\epsilon_p Baseline", color="orange")
    ax.plot(np.asarray(t_ode), eps_p_ode, label="\epsilon_p ODE",
               c="orange")
    # ax.scatter(np.asarray(t_ode)*cycl.total_time, L_ode, label="ODE",
    #           c="red")
    ax.set_xlabel("cycles")
    ax.set_ylabel("\epsilon")
    plt.legend()
    plt.show()

    # Plot LLI_sei
    fig, ax = plt.subplots()
    ax.scatter(np.asarray(t_ode), LLI, label="ODE", color="red")
    ax.legend()
    ax.set_xlabel("cycles")
    plt.show()

    # Plot Q
    fig, ax = plt.subplots()
    ax.scatter(cycles, all_Qs, label="Baseline", color="blue")
    ax.plot(np.asarray(t_ode), Q_ode, label="ODE", color="red")
    ax.legend()
    ax.set_xlabel("cycles")
    plt.show()
