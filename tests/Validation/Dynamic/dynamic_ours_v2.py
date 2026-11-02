import time
from matplotlib import pyplot as plt
import copy
from copy import deepcopy
import pickle

from main import FastAging_Simulation
from utils.param_to_inputvars import load_params
from utils.cycler import CCStep, CCCVStep, Cycler, DynamicProfile
from utils.characterize import *
from utils.pybamm_submodels import *

import scienceplots
plt.style.use(['science'])

def extract_stoich_from_param(extrap_vars, prev_params):
    model = pybamm.BaseModel()
    x_100 = pybamm.Variable("x_100")
    y_100 = pybamm.Variable("y_100")
    x_0 = pybamm.Variable("x_0")
    y_0 = pybamm.Variable("y_0")

    F = 96485.33
    eps_n = extrap_vars[3]
    Qn = (eps_n * prev_params["Negative electrode thickness [m]"] *
          prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
          prev_params["Maximum concentration in negative electrode [mol.m-3]"] * F / 3600.)

    eps_p = extrap_vars[4]
    Qp = (eps_p * prev_params["Positive electrode thickness [m]"] *
          prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
          prev_params["Maximum concentration in positive electrode [mol.m-3]"] * F / 3600.)

    n_Li = extrap_vars[0]
    U_p = prev_params["Positive electrode OCP [V]"]
    U_n = prev_params["Negative electrode OCP [V]"]

    model.algebraic = {
        x_100: U_p(y_100) - U_n(x_100) - 4.2,
        y_100: n_Li - 3600 / F * (x_100 * Qn + y_100 * Qp),
        x_0: Qn * (x_100 - x_0) - Qp * (y_0 - y_100),
        y_0: U_p(y_0) - U_n(x_0) - 2.5,
    }
    model.variables = {"x_100": x_100, "y_100": y_100, "x_0": x_0, "y_0": y_0}
    model.initial_conditions = {x_100: 0.5, y_100: 0.5, x_0: 0.5, y_0: 0.5}

    sim = pybamm.Simulation(model, parameter_values=prev_params)
    sol = sim.solve([0])
    return sol["x_0"].data[0], sol["y_0"].data[0], sol["x_100"].data[0], sol["y_100"].data[0]

def get_capacity_from_sol(y, p):
    x_0, _, x_100, _ = extract_stoich_from_param(y, p)
    eps_n = y[3]
    L = p["Negative electrode thickness [m]"]
    A = p["Electrode height [m]"] * p["Electrode width [m]"]
    cs_max = p["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    return eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600


if __name__ == "__main__":

    tot_cycles = 10000
    initial_soc = 1.
    save_fig = True

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

    # -------------- Proposed solution ---------------------

    input_vars = load_params("../../../input/M50_okane_modified_plating.yaml")
    #input_vars["D_ec"] /= 150
    # input_vars["k_sei"] /= 1000
    # input_vars["Beta_n"] *= 20
    # input_vars["Beta_p"] /= 2000
    # input_vars["k_plating"] /= 100
    input_vars["D_n"] *= 1
    # input_vars["U_sei"] = 0.4

    Q0 = input_vars["Q"]
    Q = copy.deepcopy(Q0)
    R = input_vars["R_int"]
    OCV_fn = get_OVC_func(input_vars)

    steps = [
        DynamicProfile(V_lim=[2.5, 4.2], cycle=True, dir="../../../Dynamic_Profiles/WLTC.txt"),
        CCCVStep(I=1.5, V_end=4.2, Cutoff_I=0.25, Vlim=[2.5, 4.2])
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

    start_ode_time = time.time()
    history_ode = FastAging_Simulation(
        y0_vec=y0_vec,
        input_vars=input_vars,
        tot_cycles=tot_cycles,
        stop_Q=0.1,
        cycler=cycl,
        rtol=[1e-5, 1e-3, 1e-3, 1e-5],
        atol=[1e-9, 1e-4, 1e-4, 1e-9],
        stop_condition_fn=None,  # optional early-stop
        max_step=None
    )
    ode_time = time.time() - start_ode_time
    print("ODE + integration time: ", ode_time)

    # Process results

    t_ode = history_ode["t [s]"] / 3600.
    cycles_ode = history_ode["n_cycles"]
    Lsei_ode = history_ode["y"][:, 0] * 1e9
    eps_n_ode = history_ode["y"][:, 1]
    LAMn_ode = (1 - eps_n_ode / prev_params["Negative electrode active material volume fraction"]) * 100.
    eps_p_ode = history_ode["y"][:, 2]
    LAMp_ode = (1 - eps_p_ode / prev_params["Positive electrode active material volume fraction"]) * 100.
    Lpl_ode = history_ode["y"][:, 3] * 1e9
    t_final_evol = history_ode["times"]
    Q_ode = history_ode["Q"][:]
    stoich_ode = np.asarray(history_ode["stoichiometry"])

    with open("simulation_history_WLTC_sulzer_v2.pkl", "rb") as f:
        sulzer = pickle.load(f)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    prev_param = sulzer["prev_params"]
    sulzer = sulzer["history"]
    Q_sulzer = [get_capacity_from_sol(y, prev_param) for (_, y, _) in sulzer]
    eps_p_sulzer = np.asarray([y[4] for (_, y, _) in sulzer])
    LAMp_sulzer = (1 - eps_p_sulzer / prev_params["Positive electrode active material volume fraction"]) * 100.
    eps_n_sulzer = np.asarray([y[3] for (_, y, _) in sulzer])
    LAMn_sulzer = (1 - eps_n_sulzer / prev_params["Negative electrode active material volume fraction"]) * 100.
    Lpl_sulzer = np.asarray([y[2] for (_, y, _) in sulzer])*1e9
    Lsei_sulzer = np.asarray([y[1] for (_, y, _) in sulzer])*1e9
    cycles_sulzer = [t for (t, _, _) in sulzer]

    max_ode = cycles_ode[-1]
    cutoff_idx = next(i for i, c in enumerate(cycles_sulzer) if c > max_ode)
    cutoff_idx += 1
    print(cutoff_idx + 1)
    cycles_sulzer = cycles_sulzer[:cutoff_idx]
    Q_sulzer = Q_sulzer[:cutoff_idx]
    eps_p_sulzer = eps_p_sulzer[:cutoff_idx]
    LAMp_sulzer = LAMp_sulzer[:cutoff_idx]
    eps_n_sulzer = eps_n_sulzer[:cutoff_idx]
    LAMn_sulzer = LAMn_sulzer[:cutoff_idx]
    Lpl_sulzer = Lpl_sulzer[:cutoff_idx]
    Lsei_sulzer = Lsei_sulzer[:cutoff_idx]

    xlim_cycles = cycles_ode[-1]


    fig, ax = plt.subplots()
    ax.plot(cycles_sulzer, Q_sulzer, color="black", linestyle="--")
    ax.scatter(cycles_sulzer, Q_sulzer, label="Sulzer et al.", color="black", marker="x")
    ax.scatter(cycles_ode, Q_ode, label="Proposed", color="black")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$Q [Ah]$")
    plt.legend(fontsize='x-small')
    ax.set_xlim([0, xlim_cycles])
    ax.relim(visible_only=True)
    ax.autoscale(axis='y', tight=False)
    plt.tight_layout()
    if save_fig:
        plt.savefig('../CC/Results/Q_dynamic.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(cycles_sulzer, eps_n_sulzer, color="blue", linestyle="--")
    ax.scatter(cycles_sulzer, eps_n_sulzer, label="Sulzer et al.", color="blue", marker="x")
    ax.scatter(cycles_ode, eps_n_ode, label="Proposed", color="blue")
    ax.set_xlabel("Cycles")
    ax.set_ylabel(r"$\epsilon$")
    ax.plot(cycles_sulzer, eps_p_sulzer, color="orange", linestyle="--")
    ax.scatter(cycles_sulzer, eps_p_sulzer, color="orange", marker="x")
    ax.scatter(cycles_ode, eps_p_ode, color="orange")
    ax.set_xlabel("Cycles")
    plt.legend(fontsize='x-small')
    ax.set_xlim([0, xlim_cycles])
    ax.relim(visible_only=True)
    ax.autoscale(axis='y', tight=False)
    plt.tight_layout()
    if save_fig:
        plt.savefig('../CC/Results/eps_dynamic.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(cycles_sulzer, Lpl_sulzer, color="green", linestyle="--")
    ax.scatter(cycles_sulzer, Lpl_sulzer, label="Sulzer et al.", color="green", marker="x")
    ax.scatter(cycles_ode, Lpl_ode, label="Proposed", color="green")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$L_{Plating} [nm]$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, xlim_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('../CC/Results/Lplating_dynamic.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(cycles_sulzer, Lsei_sulzer, color="lightgreen", linestyle="--")
    ax.scatter(cycles_sulzer, Lsei_sulzer, label="Sulzer et al.", color="lightgreen", marker="x")
    ax.scatter(cycles_ode, Lsei_ode, label="Proposed", color="lightgreen")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$L_{SEI} [nm]$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, xlim_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('../CC/Results/Lsei_dynamic.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(cycles_sulzer, LAMn_sulzer, color="blue", linestyle="--")
    ax.scatter(cycles_sulzer, LAMn_sulzer, label="[X]", color="blue", marker="x")
    ax.scatter(cycles_ode, LAMn_ode, label="Proposed", color="blue")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$LAM_n [\%]$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, xlim_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('../CC/Results/LAMn_dynamic.pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(cycles_sulzer, LAMp_sulzer, color="orange", linestyle="--")
    ax.scatter(cycles_sulzer, LAMp_sulzer, label="[X]", color="orange", marker="x")
    ax.scatter(cycles_ode, LAMp_ode, label="Proposed", color="orange")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("$LAM_p [\%]$")
    plt.legend(fontsize='x-small')
    plt.xlim([0, xlim_cycles])
    plt.tight_layout()
    if save_fig:
        plt.savefig('../CC/Results/LAMp_dynamic.pdf', dpi=300)
    plt.show()