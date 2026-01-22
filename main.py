from scipy.integrate import RK23
import numpy as np

from utils.degradation_models import get_dif
from utils.characterize import updated_model_Q, updated_model_R


def scipy_rk23_integrate(y0_vec, input_vars,
                         tot_cycles, cycler, stop_Q=0.2, rtol=1e-2, atol=1e-2,
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

    t_final = float(tot_cycles) * cycler.total_time * 3600.
    if max_step is None:
        max_step = t_final

    solver = RK23(rhs, t0=0.0, y0=y0_vec, t_bound=t_final,
                  max_step=max_step, rtol=rtol, atol=atol)

    n_cycles = 0.
    Q0 = cycler.Q
    Q = cycler.Q
    history = {"t [s]": [], "n_cycles": [], "y": [], "times": [], "LLI": [], "Q": [], "R": [], "soc": []}
    history["t [s]"].append(0.)
    history["n_cycles"].append(n_cycles)
    history["y"].append(y0_vec)
    history["times"].append(cycler.times)
    history["LLI"].append(0.)
    history["Q"].append(Q0)
    history["R"].append(cycler.R)
    history["soc"].append(cycler.soc0)

    #history = [(0.0, n_cycles, y0_vec, cycler.times, 0, Q0, cycler.R, cycler.soc0)]

    while solver.status == "running" and n_cycles < tot_cycles and Q > Q0 * stop_Q:
        solver.step()

        t = solver.t
        y = solver.y.copy()          # full cycle solution used at this step
        try:
            L0 =  history["y"][-1][0]
            R0 = cycler.R
            Q, LLI = updated_model_Q(y, input_vars, cycler, L0)
            # Q, LLI = updated_model_Q(y, Q, eps_0, input_vars, L0, deepcopy(param))
            R = updated_model_R(y, input_vars, L0, R0)
            n_cycles += cycler.get_n_cycles(t / 3600., history["t [s]"][-1] / 3600.)

            cycler.update_cycle(R=R, Q=Q, OCV_fn=cycler.OCV)

            history["t [s]"].append(t)
            history["n_cycles"].append(n_cycles)
            history["y"].append(y)
            history["times"].append(cycler.times)
            history["LLI"].append(LLI)
            history["Q"].append(Q)
            history["R"].append(R)
            history["soc"].append(cycler.soc[-1])
            #history.append((t, n_cycles, y, cycler.times, LLI, Q, R, cycler.soc[-1]))
        except:
            print("Q_min reached")
            break

        # Optional early stopping
        if stop_condition_fn is not None:
            if stop_condition_fn(t, y):
                break

    for k, v in history.items():
        history[k] = np.array(v)
    return history
