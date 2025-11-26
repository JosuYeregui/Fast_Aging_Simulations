import pybamm
from scipy.integrate import RK23
import time

from utils.degradation_models import get_dif, cast_params_for_SEI
from utils.cycler import CCStep, CCCVStep, Cycler
from utils.characterize import *
from matplotlib import pyplot as plt


def extract_y0_yend(sol):
    res = extract_y_from_solution(sol)
    return {k: v[0] for k, v in res.items()}, {k: v[-1] for k, v in res.items()}

def extract_y_from_solution(sol):

    LLI_sei = sol["Loss of capacity to negative SEI [A.h]"].entries
    L_sei = sol["X-averaged negative SEI thickness [m]"].entries
    eps_n = sol["X-averaged negative electrode active material volume fraction"].entries
    eps_p = sol["X-averaged positive electrode active material volume fraction"].entries

    return {"LLI_sei": LLI_sei, "L_sei": L_sei, "eps_n": eps_n, "eps_p": eps_p}


def scipy_rk23_integrate(y0_vec, input_vars,
                         t_final, cycler, rtol=1e-2, atol=1e-2,
                         stop_condition_fn=None, max_step=None):
    """
    Integrates dy/dt = Δy(y) from t=0 to t=t_final using SciPy's RK23.
    One RHS evaluation = one cycle simulation.
    Returns list of (t, y_vec, full_cycle_solution)
    """

    # Wrapper converting SciPy call signature f(t,y) → dy/dt
    def rhs(t, y):
        dy = get_dif(y, t, cycler, input_vars=input_vars)
        # store last computed cycle
        return dy

    if max_step is None:
        max_step = t_final

    solver = RK23(rhs, t0=0.0, y0=y0_vec, t_bound=t_final,
                  max_step=max_step, rtol=rtol, atol=atol)

    history = [(0.0, y0_vec[0])]

    while solver.status == "running":
        solver.step()

        t = solver.t
        y = solver.y.copy()          # full cycle solution used at this step

        history.append((t, y[0]))

        # Optional early stopping
        if stop_condition_fn is not None:
            if stop_condition_fn(t, y):
                break

    return history


if __name__ == "__main__":

    tot_cycles = 1000

    param = pybamm.ParameterValues("Chen2020")

    experiment_baseline = pybamm.Experiment(
        [(
            "Charge at 1.5 A until 3.5 V",
            "Charge at 3 A until 4.2 V",
            "Hold at 4.2 V until 0.2 A",
            "Discharge at 5 A until 3 V",)
        ] * tot_cycles
    )

    model = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"})
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

    y0_vec = [L0]

    input_vars = {}
    input_vars = cast_params_for_SEI(param, input_vars)

    start_ode_time = time.time()
    history = scipy_rk23_integrate(
        y0_vec=y0_vec,
        input_vars=input_vars,
        t_final=float(tot_cycles) * cycl.total_time * 3600.,  # e.g. 1000
        cycler=cycl,
        rtol=1e-9,
        atol=1e-9,
        stop_condition_fn=None,  # optional early-stop
        max_step=None
    )
    ode_time = time.time() - start_ode_time
    print("ODE + integration time: ", ode_time)

    t_ode = [h[0] for h in history]
    L_ode = [h[1] for h in history]

    L_bs = []
    t_baseline = []
    for s in sol_baseline.cycles:
        y, _ = extract_y0_yend(s)
        L_bs.append(y["L_sei"])
        t_baseline.append(s["Time [h]"].entries[0])

    fig, ax = plt.subplots()
    ax.plot(t_baseline, L_bs, label="Baseline")
    ax.scatter(np.asarray(t_ode) / 3600., L_ode, label="ODE",
               c="red")
    #ax.scatter(np.asarray(t_ode)*cycler.total_time, L_ode, label="ODE",
    #           c="red")
    ax.set_xlabel("t [h]")
    ax.set_ylabel("L SEI [m]")
    plt.legend()
    plt.show()