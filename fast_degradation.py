import pybamm
from matplotlib import pyplot as plt
import os
from copy import deepcopy
import time
import numpy as np

from scipy.integrate import RK23
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_capacity_from_sol(solution, param, idx, esoh_solver):
    x_0, _, x_100, _ = extract_stoich_from_sol(solution, idx, esoh_solver)
    eps_n = solution["X-averaged negative electrode active material volume fraction"].entries[idx]
    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    return eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600

def extract_stoich_from_sol(solution, idx, esoh_solver):

    Qn = solution["Negative electrode capacity [A.h]"].entries[idx]
    Qp = solution["Positive electrode capacity [A.h]"].entries[idx]
    Qp_0 = solution["X-averaged positive particle concentration"].entries[-1][idx] * \
           solution["Positive electrode capacity [A.h]"].entries[idx]
    Qn_0 = solution["X-averaged negative particle concentration"].entries[-1][idx] * \
           solution["Negative electrode capacity [A.h]"].entries[idx]
    QLi = (Qp_0 + Qn_0)

    inputs = {"V_min": 3, "V_max": 4.2,
              "Q_n": Qn,
              "Q_p": Qp,
              "Q_Li": QLi}

    esoh_sol = esoh_solver.solve(inputs)

    x_0 = esoh_sol["x_0"]
    y_0 = esoh_sol["y_0"]
    x_100 = esoh_sol["x_100"]
    y_100 = esoh_sol["y_100"]

    return x_0, y_0, x_100, y_100

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
                         t_final, rtol=1e-2, atol=1e-2,
                         stop_condition_fn=None, max_step=None):
    """
    Integrates dy/dt = Δy(y) from t=0 to t=t_final using SciPy's RK23.
    One RHS evaluation = one cycle simulation.
    Returns list of (t, y_vec, full_cycle_solution)
    """

    # Wrapper converting SciPy call signature f(t,y) → dy/dt
    def rhs(t, y):
        dy = eval_delta_y(y, t, input_vars=input_vars)
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

def eval_delta_y(L0, t, input_vars):

    L = integration(L0[0], t, input_vars)

    return L - L0

def integration(L0, t, input_vars):
    """
    Piecewise integration for dL/dt = A*D*k/(D + k*L)
    treating nearly-constant k segments analytically.
    """
    k_exp_t = input_vars["k_exp_t"]
    D = input_vars["D"]
    A = input_vars["A"]

    dt = 1.

    # Normal small-step exact update with local k
    k_mid = 0.5 * (k_exp_t(t) + k_exp_t(t + dt))
    L= (-D + np.sqrt(D**2 + 2*k_mid*(D*L0 + 0.5*k_mid*L0**2 + A*D*k_mid*dt))) / k_mid

    return L