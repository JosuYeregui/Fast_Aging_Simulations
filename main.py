from scipy.integrate import RK23
from utils.characterize import get_Q, get_Qn, get_OVC_func
from copy import deepcopy
import time

from utils.degradation_models import get_dif


def updated_model_Q(y, input_vars, cycler, L0, p):
    soc_ave = cycler.get_soc_ave()
    p = p.set_initial_stoichiometries(soc_ave)

    # Get LLI from SEI thickness
    Qn = get_Qn(p, y[1])
    F = 96485.33
    A = p["Electrode height [m]"] * p["Electrode width [m]"]
    a_s = y[1] * 3 / input_vars["Rpart_n"]
    L = p["Negative electrode thickness [m]"]
    LLI = (y[0] - L0) * A * F / (input_vars["V_SEI"] * 3600.) * input_vars["z_SEI"] * a_s * L

    # Update parameters
    p["Initial concentration in negative electrode [mol.m-3]"] -= LLI / Qn * p["Maximum concentration in negative electrode [mol.m-3]"]
    p["Initial SEI thickness [m]"] = y[0]
    p["Negative electrode active material volume fraction"] = y[1]
    p["Positive electrode active material volume fraction"] = y[2]

    # From parameters, get Q
    Q = get_Q(p)

    return Q, LLI

def updated_model_R(y, input_vars, L0, R0):

    a_s =  y[1] * 3 / input_vars["Rpart_n"]
    F = 96485.33

    R_dif = (y[0] - L0) * input_vars["rho_SEI"] / a_s * F
    R = R0 + R_dif

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

    history = [(0.0, y0_vec, cycler.times, 0, get_Q(param))]

    L0 = y0_vec[0]
    eps_0 = y0_vec[1]
    R0 = cycler.R
    Q = cycler.Q
    LLI = 0
    #param_copy = deepcopy(param)

    while solver.status == "running":
        solver.step()

        t = solver.t
        y = solver.y.copy()          # full cycle solution used at this step
        try:
            Q, LLI = updated_model_Q(y, input_vars, cycler, L0, deepcopy(param))
            # Q, LLI = updated_model_Q(y, Q, eps_0, input_vars, L0, deepcopy(param))
            R = updated_model_R(y, input_vars, L0, R0)
            cycler.update_cycle(R=R, Q=Q, OCV_fn=cycler.OCV)

            history.append((t, y, cycler.times, LLI, Q))
        except:
            print("Q_min reached")
            break

        # Optional early stopping
        if stop_condition_fn is not None:
            if stop_condition_fn(t, y):
                break

    return history
