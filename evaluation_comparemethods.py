import time
from matplotlib import pyplot as plt
import copy

from main import FastAging_Simulation
from utils.param_to_inputvars import load_params
from utils.cycler import CCStep, CCCVStep, Cycler, DynamicProfile, Rest
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
    x_0, y_0, x_100, y_100 = extract_stoich_from_sol(solution, idx, esoh_solver)
    eps_n = solution["X-averaged negative electrode active material volume fraction"].entries[idx]
    L = p["Negative electrode thickness [m]"]
    A = p["Electrode height [m]"] * p["Electrode width [m]"]
    cs_max = p["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    return eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600, [x_0, y_0, x_100, y_100 ]

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

    tot_cycles = 1000
    initial_soc = 1.

    param_chen = pybamm.ParameterValues("Chen2020")
    param = pybamm.ParameterValues("OKane2022")
    param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
    param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
    param["Electrolyte diffusivity [m2.s-1]"] = param["Electrolyte diffusivity [m2.s-1]"](1., 298.15)
    # param["Ratio of lithium moles to SEI moles"] = 2.

    param["EC diffusivity [m2.s-1]"] /= 150
    param["Positive electrode LAM constant proportional term [s-1]"] /= 2000.
    param["Negative electrode LAM constant proportional term [s-1]"] *= 20.
    param["Lithium plating kinetic rate constant [m.s-1]"] /= 100.
    param["Upper voltage cut-off [V]"] = 4.2

    param = update_LAM_params(param)

    experiment_baseline = pybamm.Experiment(
        [(
            "Rest for 24 hours",)
        ] * tot_cycles
    )

    experiment_baseline = pybamm.Experiment(
        [
            "Discharge at 1.5 A until 2.8 V",
            "Rest for 24 hour",
            "Charge at 1.5 A until 4.2 V",
            "Hold at 4.2 V until 0.2 A",

        ] * tot_cycles
    )

    experiment_baseline = pybamm.Experiment(
        [("Discharge at 1.5C until 3 V", "Rest for 1 hour", "Charge at 0.3C until 4.2 V",
          "Hold at 4.2V until C/50")] * tot_cycles)  # Sulzerren test-a

    # experiment_baseline = pybamm.Experiment(
    #     [(
    #         "Charge at 1.5 A until 3.5 V",
    #         "Charge at 1 A until 4.2 V",
    #         "Hold at 4.2 V until 0.2 A",
    #         "Rest for 1 hour",
    #         "Discharge at 1 A until 3 V",)
    #     ] * tot_cycles
    # )

    # model = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"}, build=False)
    model = pybamm.lithium_ion.SPMe(
        options={"SEI": "ec reaction limited", "x-average side reactions": "false"}, # "lithium plating": "irreversible"
        build=False)

    # model.submodels["Negative active material"] = StressBasedLAM(
    #     model.param, "Negative"
    # )

    model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

    # model.submodels["Positive active material"] = StressBasedLAM(
    #     model.param, "Positive"
    # )

    model.submodels["Plating"] = Plating_FP(model.param, "negative", x_average=True, options={"lithium plating": "irreversible"})

    model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.build_model()

    sim_baseline = pybamm.Simulation(model, experiment=experiment_baseline, parameter_values=param)

    start_time = time.time()
    sol_baseline = sim_baseline.solve(initial_soc=initial_soc)
    # sol_baseline.save("solution")
    end_time = time.time()
    print("Baseline simulation time: ", end_time - start_time)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)
    Q_bs = [get_capacity_from_sol(s, param, 0, esoh_solver)[0] for s in sol_baseline.cycles]
    stoich_bs = np.asarray([get_capacity_from_sol(s, param, 0, esoh_solver)[1] for s in sol_baseline.cycles])
    t_bs = [s["Time [h]"].entries[0] for s in sol_baseline.cycles]
    #t_bs = sol_baseline["Time [h]"].entries
    #t_bs = np.arange(len(t_bs))

    # Q_bs = []
    # stoich_bs = []
    # for idx in range(len(t_bs)):
    #     q, st = get_capacity_from_sol(sol_baseline, param, idx, esoh_solver)
    #     Q_bs.append(q)
    #     stoich_bs.append(st)
    #
    # stoich_bs = np.array(stoich_bs)
    # Q_bs = np.array(Q_bs)

    #Q_bs = [get_capacity_from_sol(sol_baseline, param, idx, esoh_solver) for idx in range(len(t_bs))]
    LLI_bs = [s["Loss of capacity to negative SEI [A.h]"].entries[0] for s in sol_baseline.cycles]
    L_bs = [s["X-averaged negative SEI thickness [m]"].entries[0] for s in sol_baseline.cycles]
    eps_n_bs = [s["X-averaged negative electrode active material volume fraction"].entries[0] for s in sol_baseline.cycles]
    eps_p_bs = [s["X-averaged positive electrode active material volume fraction"].entries[0] for s in sol_baseline.cycles]
    L_sei = [s["X-averaged negative dead lithium thickness [m]"].entries[0] for s in sol_baseline.cycles]
    t_times = [s["Time [h]"].entries[-1] - s["Time [h]"].entries[0] for s in sol_baseline.cycles]

    c_s = sol_baseline[f"X-averaged negative particle surface concentration [mol.m-3]"].entries
    # c_avg = pybamm.x_average(variables[f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"])
    c_avg = sol_baseline[f"Average negative particle concentration [mol.m-3]"].entries


    input_vars = load_params("input/M50_okane_modified_plating.yaml")
    # The following modifications are included in the modified parameter set, so they are commented out here.
    # input_vars["D_ec"] /= 150
    # #input_vars["k_sei"] /= 1000
    # input_vars["Beta_n"] *= 20
    # input_vars["Beta_p"] /= 2000
    # input_vars["k_plating"] /= 100
    # input_vars["D_n"] *= 1

    # input_vars["U_sei"] = 0.4

    Q0 = input_vars["Q"]
    Q = copy.deepcopy(Q0)
    R = input_vars["R_int"]
    OCV_fn = get_OVC_func(input_vars)

    steps = [
        Rest(t=24.),
    ]
    steps = [
        CCStep(I=-1.5, V_end=2.8),
        Rest(t=24.),
        CCCVStep(I=1.5, V_end=4.2, Cutoff_I=0.2, Vlim=[2.5, 4.2]),
    ]

    steps = [
        CCStep(I=-5*1.5, V_end=3),
        Rest(t=1.),
        #CCStep(I=5, V_end=4.2),
        CCCVStep(I=5*0.3, V_end=4.2, Cutoff_I=5/50, Vlim=[2.5, 4.2]),
    ]

    # steps = [
    #     CCStep(I=1.5, V_end=3.5),
    #     CCCVStep(I=1, V_end=4.2, Cutoff_I=0.2, Vlim=[2.5, 4.2]),
    #     Rest(t=1.),
    #     CCStep(I=-1, V_end=3.)
    # ]

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
    history = FastAging_Simulation(
        y0_vec=y0_vec,
        input_vars=input_vars,
        tot_cycles=tot_cycles,
        cycler=cycl,
        rtol=[1e-5, 1e-3, 1e-3, 1e-5],
        atol=[1e-9 ,1e-4, 1e-4, 1e-9],
        stop_condition_fn=None,  # optional early-stop
        max_step=None
    )
    ode_time = time.time() - start_ode_time
    print("ODE + integration time: ", ode_time)

    #t_ode = [h[0] for h in history]
    t_ode = history["t [s]"] / 3600.
    t_ode = history["n_cycles"]
    L_ode = history["y"][:, 0]
    eps_n_ode = history["y"][:, 1]
    eps_p_ode = history["y"][:, 2]
    plating = history["y"][:, 3]
    t_final_evol = history["times"]
    Q_ode = history["Q"]
    stoich = np.asarray(history["stoichiometry"])

    # L_bs = []
    # LLI_bs = []
    # eps_n_bs = []
    # eps_p_bs = []
    # Qlim = Q_bs[0] * 0.3
    # Q_bs = np.asarray(Q_bs)
    # Q_bs = Q_bs[Q_bs >= Qlim]
    # #t_baseline = []
    # for idx in range(len(Q_bs)):
    #     s = sol_baseline.cycles[idx]
    #     y, _ = extract_y0_yend(s)
    #     L_bs.append(y["L_sei"])
    #     LLI_bs.append(y["LLI_sei"])
    #     eps_n_bs.append(y["eps_n"])
    #     eps_p_bs.append(y["eps_p"])
        #t_baseline.append(s["Time [h]"].entries[0])

    # t_baseline = np.arange(len(eps_p_bs))
    t_baseline = np.asarray(t_bs) / t_bs[-1] * tot_cycles

    fig, ax = plt.subplots()
    ax.plot(t_baseline, L_bs, label="Baseline")
    ax.scatter(np.asarray(t_ode), L_ode, label="ODE",
               c="red")
    #ax.scatter(np.asarray(t_ode)*cycl.total_time, L_ode, label="ODE",
    #           c="red")
    ax.set_xlabel("cycles")
    ax.set_ylabel("L SEI [m]")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(t_baseline, eps_n_bs, label="Baseline", color="blue")
    ax.scatter(np.asarray(t_ode), eps_n_ode, label="epsilon_n ODE",
               c="blue")
    ax.plot(t_baseline, eps_p_bs, label="Baseline", color="orange")
    ax.scatter(np.asarray(t_ode), eps_p_ode, label="epsilon_p ODE",
               c="orange")
    # ax.scatter(np.asarray(t_ode)*cycl.total_time, L_ode, label="ODE",
    #           c="red")
    ax.set_xlabel("cycles")
    ax.set_ylabel("epsilon")
    plt.legend()
    plt.show()

    # Plot Q
    fig, ax = plt.subplots()
    ax.plot(t_baseline, Q_bs, label="Baseline", color="blue")
    ax.scatter(np.asarray(t_ode), Q_ode, label="ODE", color="red")
    ax.legend()
    ax.set_xlabel("cycles")
    plt.show()
    #
    fig, ax = plt.subplots()
    ax.plot(t_baseline, L_sei, label="Baseline", color="blue")
    ax.scatter(np.asarray(t_ode), plating, label="ODE", color="red")
    ax.legend()
    ax.set_xlabel("cycles")
    plt.show()
    #
    fig, ax = plt.subplots()
    ax.plot(t_baseline, stoich_bs[:, 0], label="Baseline x0", color="blue")
    ax.plot(t_baseline, stoich_bs[:, 2], label="Baseline x100", color="cyan")
    ax.scatter(np.asarray(t_ode), stoich[:, 0], label="ODE x0", color="blue")
    ax.scatter(np.asarray(t_ode), stoich[:, 1], label="ODE x100", color="cyan")
    ax.plot(t_baseline, stoich_bs[:, 1], label="Baseline y0", color="orange")
    ax.plot(t_baseline, stoich_bs[:, 3], label="Baseline y100", color="red")
    ax.scatter(np.asarray(t_ode), stoich[:, 2], label="ODE y0", color="orange")
    ax.scatter(np.asarray(t_ode), stoich[:, 3], label="ODE y100", color="red")
    ax.legend()
    ax.set_xlabel("cycles")
    ax.set_ylabel("Stoichiometry")
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(t_baseline, t_times)
    ax.plot(np.asarray(t_ode), t_final_evol)
    ax.legend()
    ax.set_xlabel("cycles")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(c_s - c_avg)
    ax.legend()
    ax.set_xlabel("cycles")
    plt.show()
