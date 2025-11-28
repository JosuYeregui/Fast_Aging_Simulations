import time
from matplotlib import pyplot as plt

from main import scipy_rk23_integrate
from utils.degradation_models import cast_params_for_SEI, cast_params_for_LAM
from utils.cycler import CCStep, CCCVStep, Cycler
from utils.characterize import *
from utils.pybamm_submodels import *

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

    param = update_LAM_params(param)

    experiment_baseline = pybamm.Experiment(
        [(
            "Charge at 1.5 A until 3.5 V",
            "Charge at 3 A until 4.2 V",
            "Hold at 4.2 V until 0.2 A",
            "Discharge at 5 A until 3 V",)
        ] * tot_cycles
    )

    model = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"}, build=False)

    model.submodels["Negative active material"] = StressBasedLAM(
        model.param, "Negative"
    )

    # model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.submodels["Positive active material"] = StressBasedLAM(
        model.param, "Positive"
    )

    model.build_model()

    sim_baseline = pybamm.Simulation(model, experiment=experiment_baseline, parameter_values=param)

    start_time = time.time()
    sol_baseline = sim_baseline.solve(initial_soc=0.1)
    end_time = time.time()
    print("Baseline simulation time: ", end_time - start_time)

    Q = get_Q(param)
    R = get_Rint(model, param)
    OCV_fn = get_OVC_func(param)

    steps = [
        CCStep(I=1.5, V_end=3.5),
        CCCVStep(I=3, V_end=4.2, Cutoff_I=0.2),
        CCStep(I=-5, V_end=3.)
    ]

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

    L_bs = []
    eps_n_bs = []
    eps_p_bs = []
    t_baseline = []
    for s in sol_baseline.cycles:
        y, _ = extract_y0_yend(s)
        L_bs.append(y["L_sei"])
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