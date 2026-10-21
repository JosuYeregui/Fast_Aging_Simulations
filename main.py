from scipy.integrate import RK23
import numpy as np
import copy

from utils.degradation_models import get_dif
from utils.characterize import updated_model_Q, updated_model_R


def FastAging_Simulation(y0_vec, input_vars,
                         tot_cycles, cycler, stop_Q=0.2, rtol=1e-2, atol=1e-2,
                         stop_condition_fn=None, max_step=None, return_system_vars=False, print_Qmin=True):
    """
    Integrates dy/dt = Δy(y) from t=0 to t=t_final using SciPy's RK23.
    One RHS evaluation = one cycle simulation.
    Returns list of (t, y_vec, full_cycle_solution)
    """

    system_vars = copy.deepcopy(input_vars)
    system_cycler = copy.deepcopy(cycler)

    # Wrapper converting SciPy call signature
    def rhs(t, y):
        L0 = system_vars["L0_sei"]
        L0_pl = system_vars["L0_pl"]

        R0 = system_cycler.R
        Q, LLI = updated_model_Q(y, system_vars, system_cycler, L0, L0_pl, return_lli=True)
        # Q, LLI = updated_model_Q(y, Q, eps_0, input_vars, L0, deepcopy(param))
        R = updated_model_R(y, system_vars, L0, R0)

        system_cycler.update_cycle(R=R, Q=Q, OCV_fn=system_cycler.OCV)

        dy = get_dif(y, t, system_cycler, input_vars=system_vars)

        system_vars["L0_sei"] = y[0]
        system_vars["L0_pl"] = y[3]
        system_vars["LLI"] = LLI * 100.
        # store last computed cycle
        return dy

    t_final = float(tot_cycles) * system_cycler.total_time * 3600.
    if max_step is None:
        max_step = t_final

    #L0 = y0_vec[0]
    #L0_pl = y0_vec[3]

    system_vars["L0_sei"] = y0_vec[0]
    system_vars["L0_pl"] = y0_vec[3]
    system_vars["LLI"] = 0.

    solver = RK23(rhs, t0=0.0, y0=y0_vec, t_bound=t_final,
                  max_step=max_step, rtol=rtol, atol=atol)

    n_cycles = 0.
    Q0 = system_cycler.Q
    Q = system_cycler.Q
    history = {"t [s]": [], "n_cycles": [], "y": [], "times": [], "Q": [], "R": [], "soc": [], "stoichiometry": [], "modes": []}
    history["t [s]"].append(0.)
    history["n_cycles"].append(n_cycles)
    history["y"].append(y0_vec)
    history["times"].append(system_cycler.times)
    history["Q"].append(Q0)
    history["R"].append(system_cycler.R)
    history["soc"].append(system_cycler.soc0)
    history["stoichiometry"].append([system_vars["x_0"], system_vars["x_100"], system_vars["y_0"], system_vars["y_100"]])
    history["modes"].append([y0_vec[1] / input_vars["eps_0_n"]*100., y0_vec[2] / input_vars["eps_0_p"]*100., system_vars["LLI"]])  # initial mode values

    #history = [(0.0, n_cycles, y0_vec, cycler.times, 0, Q0, cycler.R, cycler.soc0)]

    while solver.status == "running" and n_cycles < tot_cycles and Q > Q0 * stop_Q:

        try:
            solver.step()
        except:
            if print_Qmin:
                print("Q_min reached")
            break

        t = solver.t
        y = solver.y.copy()

        n_cycles += system_cycler.get_n_cycles(t / 3600., history["t [s]"][-1] / 3600.)# full cycle solution used at this step

        history["t [s]"].append(t)
        history["n_cycles"].append(n_cycles)
        history["y"].append(y)
        history["times"].append(system_cycler.times)
        history["Q"].append(system_cycler.Q)
        history["R"].append(system_cycler.R)
        history["soc"].append(system_cycler.soc[-1])
        history["stoichiometry"].append(
            (system_vars["x_0"], system_vars["x_100"], system_vars["y_0"], system_vars["y_100"]))
        history["modes"].append([y[1] / input_vars["eps_0_n"] * 100., y[2] / input_vars["eps_0_p"] * 100.,
                                 system_vars["LLI"]])
        Q = history["Q"][-1]

        # Optional early stopping
        if stop_condition_fn is not None:
            if stop_condition_fn(t, y):
                break

    for k, v in history.items():
        history[k] = np.array(v)

    if return_system_vars:
        return history, system_vars, system_cycler

    return history