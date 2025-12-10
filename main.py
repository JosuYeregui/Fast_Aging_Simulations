from scipy.integrate import RK23
from utils.characterize import get_Q, get_Qn, stoichiometric_window
from copy import deepcopy
import time

from utils.degradation_models import get_dif


def updated_model_Q(y, input_vars, cycler, L0):
    soc_ave = cycler.get_soc_ave()
    sol_n = soc_ave * (input_vars["x_100"] - input_vars["x_0"]) + input_vars["x_0"]
    input_vars["cini_n_0"] = sol_n * input_vars["csmax_n"]
    sol_p = soc_ave * (input_vars["y_100"] - input_vars["y_0"]) + input_vars["y_0"]
    input_vars["cini_p_0"] = sol_p * input_vars["csmax_p"]
    #p = p.set_initial_stoichiometries(soc_ave)

    # Get LLI from SEI thickness
    Qn = get_Qn(input_vars, y[1])
    F = 96485.33
    A = input_vars["A"]
    a_s = y[1] * 3 / input_vars["Rpart_n"]
    L = input_vars["L_n"]
    LLI = (y[0] - L0) * A * F / (input_vars["V_SEI"] * 3600.) * input_vars["z_SEI"] * a_s * L

    # Update parameters
    input_vars["cini_n"] = input_vars["cini_n_0"] - LLI / Qn * input_vars["csmax_n"]
    input_vars["cini_p"] = input_vars["cini_p_0"]

    x_0, x_100, y_100, y_0 = stoichiometric_window(input_vars, y)
    input_vars["x_0"] = x_0
    input_vars["x_100"] = x_100
    input_vars["y_0"] = y_0
    input_vars["y_100"] = y_100

    # From parameters, get Q
    Q = get_Q(input_vars, y)

    return Q, LLI

def updated_model_R(y, input_vars, L0, R0):

    a_s =  y[1] * 3 / input_vars["Rpart_n"]
    F = 96485.33

    R_dif = (y[0] - L0) * input_vars["rho_SEI"] / a_s * F
    R = R0 + R_dif

    return R


def scipy_rk23_integrate(y0_vec, input_vars,
                         tot_cycles, cycler, rtol=1e-2, atol=1e-2,
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
    history = [(0.0, n_cycles, y0_vec, cycler.times, 0, get_Q(input_vars, y0_vec))]

    while solver.status == "running" and n_cycles < tot_cycles:
        solver.step()

        t = solver.t
        y = solver.y.copy()          # full cycle solution used at this step
        try:
            L0 =  history[-1][2][0]
            R0 = cycler.R
            Q, LLI = updated_model_Q(y, input_vars, cycler, L0)
            # Q, LLI = updated_model_Q(y, Q, eps_0, input_vars, L0, deepcopy(param))
            R = updated_model_R(y, input_vars, L0, R0)
            n_cycles += cycler.get_n_cycles(t / 3600., history[-1][0] / 3600.)

            cycler.update_cycle(R=R, Q=Q, OCV_fn=cycler.OCV)

            history.append((t, n_cycles, y, cycler.times, LLI, Q))
        except:
            print("Q_min reached")
            break

        # Optional early stopping
        if stop_condition_fn is not None:
            if stop_condition_fn(t, y):
                break

    return history
