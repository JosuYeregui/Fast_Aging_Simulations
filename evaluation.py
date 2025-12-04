import time
from matplotlib import pyplot as plt

from main import scipy_rk23_integrate
from utils.degradation_models import cast_params_for_SEI, cast_params_for_LAM
from utils.cycler import CCStep, CCCVStep, Cycler
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

    tot_cycles = 100

    param_chen = pybamm.ParameterValues("Chen2020")
    param = pybamm.ParameterValues("OKane2022")
    param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
    param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
    # param["Ratio of lithium moles to SEI moles"] = 2.

    param["EC diffusivity [m2.s-1]"] /= 10
    param["Positive electrode LAM constant proportional term [s-1]"] /= 100.
    param["Negative electrode LAM constant proportional term [s-1]"] *= 200.

    param = update_LAM_params(param)

    experiment_baseline = pybamm.Experiment(
        [(
            "Charge at 1.5 A until 3.5 V",
            "Charge at 1 A until 4.2 V",
            "Hold at 4.2 V until 0.2 A",
            "Discharge at 1 A until 3 V",)
        ] * tot_cycles
    )

    model = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"}, build=False)

    # model.submodels["Negative active material"] = StressBasedLAM(
    #     model.param, "Negative"
    # )

    model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

    # model.submodels["Positive active material"] = StressBasedLAM(
    #     model.param, "Positive"
    # )

    model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.build_model()

    sim_baseline = pybamm.Simulation(model, experiment=experiment_baseline, parameter_values=param)

    start_time = time.time()
    sol_baseline = sim_baseline.solve(initial_soc=0.1)
    end_time = time.time()
    print("Baseline simulation time: ", end_time - start_time)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)
    Q_bs = [get_capacity_from_sol(s, param, 0, esoh_solver) for s in sol_baseline.cycles]

    Q = get_Q(param)
    R = get_Rint(model, param)
    OCV_fn = get_OVC_func(param)

    steps = [
        CCStep(I=1.5, V_end=3.5),
        CCCVStep(I=1, V_end=4.2, Cutoff_I=0.2),
        CCStep(I=-1, V_end=3.)
    ]
    from copy import deepcopy as copy
    cycl = Cycler(
        steps=steps,
        SOC0=0.1,
        R=R,
        Q=Q,
        OCV_fn=OCV_fn
    )

    L0 = float(param.get("initial sei thickness", 5e-9))
    eps_0_n = param["Negative electrode active material volume fraction"]
    eps_0_p = param["Positive electrode active material volume fraction"]

    y0_vec = [L0, eps_0_n, eps_0_p]

    input_vars = {}
    input_vars = cast_params_for_SEI(param, input_vars)
    input_vars = cast_params_for_LAM(param, input_vars)

    start_ode_time = time.time()
    history = scipy_rk23_integrate(
        y0_vec=y0_vec,
        input_vars=input_vars,
        t_final=float(tot_cycles) * cycl.total_time * 3600.,  # e.g. 1000
        cycler=cycl,
        param=param,
        rtol=[1e-3, 1e-3, 1e-3],
        atol=[1e-9 ,1e-4, 1e-4],
        stop_condition_fn=None,  # optional early-stop
        max_step=None
    )
    ode_time = time.time() - start_ode_time
    print("ODE + integration time: ", ode_time)

    t_ode = [h[0] for h in history]
    L_ode = [h[1][0] for h in history]
    eps_n_ode = [h[1][1] for h in history]
    eps_p_ode = [h[1][2] for h in history]
    t_final_evol = [h[2] for h in history]
    LLI = [h[3] for h in history]
    Q_ode = [h[4] for h in history]

    L_bs = []
    LLI_bs = []
    eps_n_bs = []
    eps_p_bs = []
    t_baseline = []
    for s in sol_baseline.cycles:
        y, _ = extract_y0_yend(s)
        L_bs.append(y["L_sei"])
        LLI_bs.append(y["LLI_sei"])
        eps_n_bs.append(y["eps_n"])
        eps_p_bs.append(y["eps_p"])
        t_baseline.append(s["Time [h]"].entries[0])

    fig, ax = plt.subplots()
    ax.plot(t_baseline, L_bs, label="Baseline")
    ax.scatter(np.asarray(t_ode) / 3600., L_ode, label="ODE",
               c="red")
    #ax.scatter(np.asarray(t_ode)*cycl.total_time, L_ode, label="ODE",
    #           c="red")
    ax.set_xlabel("t [h]")
    ax.set_ylabel("L SEI [m]")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(t_baseline, eps_n_bs, label="Baseline", color="blue")
    ax.scatter(np.asarray(t_ode) / 3600., eps_n_ode, label="\epsilon_n ODE",
               c="blue")
    ax.plot(t_baseline, eps_p_bs, label="Baseline", color="orange")
    ax.scatter(np.asarray(t_ode) / 3600., eps_p_ode, label="\epsilon_p ODE",
               c="orange")
    # ax.scatter(np.asarray(t_ode)*cycl.total_time, L_ode, label="ODE",
    #           c="red")
    ax.set_xlabel("t [h]")
    ax.set_ylabel("\epsilon")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.asarray(t_ode) / 3600., t_final_evol)
    # ax.scatter(np.asarray(t_ode)*cycl.total_time, L_ode, label="ODE",
    #           c="red")
    ax.set_xlabel("t [h]")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.diff(t_baseline))
    # ax.scatter(np.asarray(t_ode) / 3600. / 3, np.asarray(t_final_evol)[:, -1], label="ODE",
    #           c="red")
    ax.set_xlabel("t [h]")
    plt.show()

    # Plot LLI_sei
    fig, ax = plt.subplots()
    ax.plot(t_baseline, LLI_bs, label="Baseline", color="blue")
    ax.scatter(np.asarray(t_ode) / 3600., LLI, label="ODE", color="red")
    ax.legend()
    ax.set_xlabel("t [h]")
    plt.show()

    # Plot Q
    fig, ax = plt.subplots()
    ax.plot(t_baseline, Q_bs, label="Baseline", color="blue")
    ax.scatter(np.asarray(t_ode) / 3600., Q_ode, label="ODE", color="red")
    ax.legend()
    ax.set_xlabel("t [h]")
    plt.show()