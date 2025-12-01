from scipy.integrate import RK23
from utils.characterize import get_Q, get_OVC_func
from copy import deepcopy

from utils.degradation_models import get_dif


def updated_model_Q(y, p):

    p["Initial SEI thickness [m]"] = y[0]
    p["Negative electrode active material volume fraction"] = y[1]
    p["Positive electrode active material volume fraction"] = y[2]

    Q = get_Q(p)

    return Q

def updated_model_R(y, input_vars, L0, R0):

    R_dif = (y[0] - L0) * input_vars["rho_SEI"]
    R = R0 #+ R_dif

    return R


def scipy_rk23_integrate(y0_vec, input_vars, param,
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

    history = [(0.0, y0_vec, cycler.times)]

    L0 = y0_vec[0]
    R0 = cycler.R

    while solver.status == "running":
        solver.step()

        t = solver.t
        y = solver.y.copy()          # full cycle solution used at this step
        try:
            Q = updated_model_Q(y, deepcopy(param))
            R = updated_model_R(y, input_vars, L0, R0)
            cycler.update_cycle(R=R, Q=Q, OCV_fn=cycler.OCV)

            history.append((t, y, cycler.times))
        except:
            print("Q_min reached")
            break

        # Optional early stopping
        if stop_condition_fn is not None:
            if stop_condition_fn(t, y):
                break

    return history
