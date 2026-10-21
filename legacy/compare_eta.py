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
from utils.degradation_models import get_t_and_dt


if __name__ == "__main__":

    initial_soc = 1.
    save_fig = False
    T_dev = -30


    # -------------- Define model ---------------------

    param_chen = pybamm.ParameterValues("Chen2020")
    param = pybamm.ParameterValues("OKane2022")
    param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
    param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
    #param["Electrolyte diffusivity [m2.s-1]"] = param["Electrolyte diffusivity [m2.s-1]"](1., 298.15)
    func = copy.deepcopy(param["Electrolyte diffusivity [m2.s-1]"])
    param["Electrolyte diffusivity [m2.s-1]"] = lambda c, T: func(1., T)

    param["EC diffusivity [m2.s-1]"] /= 150
    param["Positive electrode LAM constant proportional term [s-1]"] /= 2000.
    param["Negative electrode LAM constant proportional term [s-1]"] *= 20.
    param["Lithium plating kinetic rate constant [m.s-1]"] /= 100.

    param["Initial temperature [K]"] += T_dev
    param["Ambient temperature [K]"] += T_dev

    param["Upper voltage cut-off [V]"] = 4.2
    param = update_LAM_params(param)
    param.set_initial_state(initial_soc)
    prev_params = deepcopy(param)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    model = pybamm.lithium_ion.SPMe(
        options={"SEI": "ec reaction limited", "x-average side reactions": "false", "use lumped thermal capacity": "false"},
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

    # Compute baseline HERE

    # -------------- Baseline ---------------------


    exp = ("Discharge at 1C until 3 V", "Rest for 1 hour", "Charge at 0.3C until 3.6 V","Hold at 3.6V until C/50")

    experiment = pybamm.Experiment(exp)
    start_time = time.time()
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
    base_sol = sim.solve(initial_soc=initial_soc)
    print("Baseline time: ", time.time() - start_time)

    # Extract PyBaMM overpotentials for discharge (first step)
    discharge_sol = base_sol
    t_pybamm = discharge_sol["Time [s]"].entries
    eta_pybamm = discharge_sol["X-averaged negative electrode reaction overpotential [V]"].entries
    eta_pl_pybamm = discharge_sol["X-averaged negative electrode lithium plating reaction overpotential [V]"].entries
    j_pl_pybamm = discharge_sol["X-averaged negative electrode lithium plating interfacial current density [A.m-2]"].entries

    n_Li0 = (base_sol["Average negative particle concentration [mol.m-3]"].entries / param["Maximum concentration in negative electrode [mol.m-3]"] * base_sol["Negative electrode capacity [A.h]"].entries + base_sol["Average positive particle concentration [mol.m-3]"].entries / param["Maximum concentration in positive electrode [mol.m-3]"] * base_sol[
               "Positive electrode capacity [A.h]"].entries) * 3600 / 96485.33
    n_Li0 = n_Li0[0]


    input_vars = load_params("../input/M50_okane_modified_plating.yaml")
    input_vars["D_ec"] /= 150
    # input_vars["k_sei"] /= 1000
    input_vars["Beta_n"] *= 20
    input_vars["Beta_p"] /= 2000
    input_vars["k_plating"] /= 100
    input_vars["D_n"] *= 1
    # input_vars["U_sei"] = 0.4

    Q0 = input_vars["Q"]
    #input_vars["R_int"] = 0.06263196455485556
    Q = copy.deepcopy(Q0)
    R = input_vars["R_int"]
    OCV_fn = get_OVC_func(input_vars)

    steps = [
        CCStep(I=-5 * 1, V_end=3),
        Rest(t=1.),
        # CCStep(I=5, V_end=4.2),
        CCCVStep(I=5 * 0.3, V_end=3.6, Cutoff_I=5 / 50, Vlim=[2.5, 4.2]),
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

    input_vars["T"] += T_dev

    # Compute our approximation eta_r for discharge
    t_app, _ = get_t_and_dt(cycl)
    I_t_app, soc_t_app = [], []
    for t_i in t_app:
        i_val, s_val = cycl(t_i)
        I_t_app.append(-i_val)
        soc_t_app.append(s_val)

    I_raw_app = np.array(I_t_app)
    soc_t_app = np.array(soc_t_app)

    y = y0_vec
    R_gas = input_vars["R"]
    T_val = input_vars["T"]
    F_val = input_vars["F"]
    RTF_app = R_gas * T_val / F_val

    as_n_app = 3 * y[1] / input_vars["R_n"]
    I_app = I_raw_app / input_vars["A"]

    arr_De = np.exp(input_vars["E_act_De"] / R_gas * (1 / 298.15 - 1 / T_val))
    Deff_n_app = input_vars["D_e"] * arr_De * input_vars["por_n"] ** input_vars["brug_n"]

    ce_int_app = (input_vars["ce_0"] + I_app * (1 - input_vars["t_plus"]) / F_val *
                  (input_vars["L_n"] / (3 * Deff_n_app)))
    ce_int_app_2 = discharge_sol["X-averaged negative electrolyte concentration [mol.m-3]"].entries

    #fig, ax = plt.subplots()
    #ax.plot(np.linspace(0, 1, len(ce_int_app)), ce_int_app)
    #ax.plot(np.linspace(0, 1, len(ce_int_app_2)), ce_int_app_2)
    #plt.show()

    sol_n_app = soc_t_app * (input_vars["x_100"] - input_vars["x_0"]) + input_vars["x_0"]
    sol_n_app += input_vars["cs_cave_n"](I_raw_app, y[1]) / input_vars["csmax_n"]

    arr_j0 = np.exp(input_vars["E_act_j0"] / R_gas * (1 / 298.15 - 1 / T_val))
    j0_app = (np.sqrt(ce_int_app) * np.sqrt(sol_n_app * input_vars["csmax_n"]) *
              np.sqrt(input_vars["csmax_n"] - sol_n_app * input_vars["csmax_n"]) *
              input_vars["m_ref"] * arr_j0)

    eta_r_app = -2 * RTF_app * np.arcsinh(I_app / (as_n_app * j0_app * input_vars["L_n"]))

    U_n_vec = np.vectorize(input_vars["U_n"])
    ocp_app = U_n_vec(sol_n_app)
    eta_pl_app = ocp_app - eta_r_app

    i0_app = input_vars["k_plating"] * ce_int_app * F_val
    j_pl_app = -i0_app * np.exp(-input_vars["alpha_plating"] * eta_pl_app / RTF_app)

    # Plot reaction overpotential comparison
    # fig, ax = plt.subplots()
    # ax.plot(t_pybamm - t_pybamm[0], eta_pybamm, label="PyBaMM")
    # ax.plot(t_app*3600., -eta_r_app, label="Approximation")
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Negative electrode reaction overpotential [V]")
    # ax.legend()
    # plt.tight_layout()

    # Plot plating overpotential comparison
    fig2, ax2 = plt.subplots()
    ax2.plot(t_pybamm - t_pybamm[0], eta_pl_pybamm, label="PyBaMM")
    ax2.plot(t_app*3600., eta_pl_app, label="Guria")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Lithium plating overpotential [V]")
    ax2.legend()
    plt.tight_layout()

    # Plot plating current density comparison
    fig3, ax3 = plt.subplots()
    ax3.plot(t_pybamm - t_pybamm[0], j_pl_pybamm, label="PyBaMM")
    ax3.plot(t_app * 3600., j_pl_app, label="Guria")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Lithium plating current density [A.m-2]")
    ax3.legend()
    plt.tight_layout()
    plt.show()
