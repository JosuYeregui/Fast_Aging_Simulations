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
        dy = eval_delta_y(y, input_vars=input_vars)
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

def eval_delta_y(L0, input_vars):

    L = piecewise_integration(L0[0], input_vars, slope_tol=1e-3, min_len=5)

    return L[-1] - L0

def piecewise_integration(L0, input_vars, slope_tol=1e-3, min_len=5):
    """
    Piecewise integration for dL/dt = A*D*k/(D + k*L)
    treating nearly-constant k segments analytically.
    """
    k_exp_t = input_vars["k_exp_t"]
    D = input_vars["D"]
    A = input_vars["A"]
    t = input_vars["t"]

    k_exp_t = np.asarray(k_exp_t)
    dt = np.diff(t)

    # Compute relative slope of k_exp_t to detect flat regions ("plateaus")
    dk = np.gradient(k_exp_t, t,)
    flat = np.abs(dk / np.maximum(np.abs(k_exp_t), 1e-30)) < slope_tol

    # Identify contiguous flat regions
    segments = []
    i = 0
    N = len(t)
    while i < N - 1:
        if flat[i]:
            j = i
            while j < N - 1 and flat[j]:
                j += 1
            if j - i >= min_len:
                segments.append((i, j))
            i = j
        else:
            i += 1

    # Integrate step by step
    L = np.zeros_like(t)
    L[0] = L0
    seg_idx = 0

    i = 0
    while i < N - 1:
        # If this point is inside a plateau segment
        if seg_idx < len(segments) and segments[seg_idx][0] <= i < segments[seg_idx][1]:
            i_start, i_end = segments[seg_idx]
            k_plateau = np.mean(k_exp_t[i_start:i_end+1])
            dt_plateau = t[i_end] - t[i_start]
            Li = L[i_start]
            # Exact constant-k update
            L_next = (-D + np.sqrt(D**2 + 2*k_plateau*(D*Li + 0.5*k_plateau*Li**2 + A*D*k_plateau*dt_plateau))) / k_plateau
            L[i_start+1:i_end+1] = L_next
            i = i_end
            seg_idx += 1
        else:
            # Normal small-step exact update with local k
            k_mid = 0.5 * (k_exp_t[i] + k_exp_t[i+1])
            L[i+1] = (-D + np.sqrt(D**2 + 2*k_mid*(D*L[i] + 0.5*k_mid*L[i]**2 + A*D*k_mid*dt[i]))) / k_mid
            # L_anal = (-D_ec + np.sqrt((D_ec + k_exp * L0)**2 + 2 * A * D_ec * k_exp ** 2 * t)) / k_exp

            i += 1
    return L

# sol_baseline.cycles[-1].steps[-1].y_slices
if __name__ == "__main__":

    tot_cycles = 1000 # Total cycles to simulate

    experiment_baseline = pybamm.Experiment(
        [(
                "Charge at 1 C until 4.2 V",
                "Hold at 4.2 V until 0.02 C",
                "Discharge at 0.5 C until 2.5 V",)
        ] * tot_cycles
    )

    param = pybamm.ParameterValues("Chen2020")
    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    # spm = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited", "loss of active material": "stress-driven","x-average side reactions":"true"})
    spm = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"})
    sim_baseline = pybamm.Simulation(spm, experiment=experiment_baseline, parameter_values=param)
    start_time = time.time()
    sol_baseline = sim_baseline.solve()
    end_time = time.time()
    print("Baseline simulation time: ", end_time - start_time)

    # Q_bs = [get_capacity_from_sol(s, param, 0, esoh_solver) for s in sol_baseline.cycles]
    # eps_bs = [s["X-averaged negative electrode active material volume fraction"].entries[-1] for s in sol_baseline.cycles]

    j0 = float(param.get("SEI reaction exchange current density [A.m-2]",
                         1.5e-7))  # or "SEI reaction exchange current density"
    k_sei = float(param.get("SEI kinetic rate constant [m.s-1]", 1e-12))
    D_ec = float(param.get("EC diffusivity [m2.s-1]", 2e-18))  # or "Molar mass of SEI"
    c_ec = float(param.get("EC initial concentration in electrolyte [mol.m-3]", 4541))
    alpha = float(param.get("SEI charge transfer coefficient", 0.5))
    U_sei = float(param.get("SEI open-circuit potential [V]", 0.4))
    F = float(param.get("Faraday constant [C.mol-1]", 96485.33212))
    R = float(param.get("Ideal gas constant [J.mol-1.K-1]", 8.314462618))
    T = float(param.get("Ambient temperature [K]", 298.15))
    L0 = float(param.get("initial sei thickness", 5e-9))
    V_sei = float(param.get("sei partial molar volume", 9.585e-5))

    y0_vec = [L0]

    t = np.asarray(range(3600 + 7200 + 2), dtype=float)
    Uocp_p = param["Positive electrode OCP [V]"]
    Uocp_n = param["Negative electrode OCP [V]"]
    SOL_n = np.append(np.linspace(0.03, 0.9, 3601), np.linspace(0.9, 0.03, 7201))

    #  = np.tile(SOL_n_cycle, n_cycles * 2)
    eta = Uocp_n(SOL_n) - U_sei

    exp_vals = np.exp(-alpha * F * eta / (R * T))
    k_exp = exp_vals * k_sei
    A = c_ec * V_sei / 2

    input_vars = {"k_exp_t": k_exp, "A": A, "D": D_ec, "t": t}

    start_ode_time = time.time()

    history = scipy_rk23_integrate(
        y0_vec=y0_vec,
        input_vars=input_vars,
        t_final=float(tot_cycles),  # e.g. 1000
        rtol=1e-9,
        atol=1e-9,
        stop_condition_fn=None,  # optional early-stop
        max_step=None
    )
    ode_time = time.time() - start_ode_time
    print("ODE + PW time: ", ode_time)

    cycles_ode = [h[0] for h in history]
    L_ode = [h[1] for h in history]

    L_bs = []
    t_baseline = []
    for s in sol_baseline.cycles:
        y, _ = extract_y0_yend(s)
        L_bs.append(y["L_sei"])
        t_baseline.append(s["Time [h]"].entries[0])

    fig, ax = plt.subplots()
    ax.plot(t_baseline, L_bs, label="Baseline")
    ax.scatter(np.asarray(cycles_ode) * (3600 + 7200 + 2)/3600, L_ode, label="ODE",
               c="red")
    #ax.scatter(list(range(len(extrapolation_results["Q"]))), extrapolation_results["Q"], label="Extrapolated N=1",
    #           c="green")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("L SEI [m]")
    plt.legend()
    plt.show()

