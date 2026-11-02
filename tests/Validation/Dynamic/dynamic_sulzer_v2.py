import pybamm
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from copy import deepcopy
import time

from scipy.integrate import RK23
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utils.pybamm_submodels import update_LAM_params, StressBasedLAM_FP, Plating_FP


def extract_stoich_from_param(extrap_vars, prev_params):
    model = pybamm.BaseModel()
    x_100 = pybamm.Variable("x_100")
    y_100 = pybamm.Variable("y_100")
    x_0 = pybamm.Variable("x_0")
    y_0 = pybamm.Variable("y_0")

    F = 96485.33
    eps_n = extrap_vars["eps_n"]
    Qn = (eps_n * prev_params["Negative electrode thickness [m]"] *
          prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
          prev_params["Maximum concentration in negative electrode [mol.m-3]"] * F / 3600.)

    eps_p = extrap_vars["eps_p"]
    Qp = (eps_p * prev_params["Positive electrode thickness [m]"] *
          prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
          prev_params["Maximum concentration in positive electrode [mol.m-3]"] * F / 3600.)

    n_Li = extrap_vars["n_Li"]
    U_p = prev_params["Positive electrode OCP [V]"]
    U_n = prev_params["Negative electrode OCP [V]"]

    model.algebraic = {
        x_100: U_p(y_100) - U_n(x_100) - 4.2,
        y_100: n_Li - 3600 / F * (x_100 * Qn + y_100 * Qp),
        x_0: Qn * (x_100 - x_0) - Qp * (y_0 - y_100),
        y_0: U_p(y_0) - U_n(x_0) - 2.5,
    }
    model.variables = {"x_100": x_100, "y_100": y_100, "x_0": x_0, "y_0": y_0}
    model.initial_conditions = {x_100: 0.5, y_100: 0.5, x_0: 0.5, y_0: 0.5}

    sim = pybamm.Simulation(model, parameter_values=prev_params)
    sol = sim.solve([0])
    return sol["x_0"].data[0], sol["y_0"].data[0], sol["x_100"].data[0], sol["y_100"].data[0]


def extract_y_from_solution(sol, prev_params):
    F = 96485.33
    eps_n = sol["X-averaged negative electrode active material volume fraction"].entries
    Qn = (eps_n * prev_params["Negative electrode thickness [m]"] *
          prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
          prev_params["Maximum concentration in negative electrode [mol.m-3]"] * F / 3600.)

    eps_p = sol["X-averaged positive electrode active material volume fraction"].entries
    Qp = (eps_p * prev_params["Positive electrode thickness [m]"] *
          prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
          prev_params["Maximum concentration in positive electrode [mol.m-3]"] * F / 3600.)

    n_Li = (sol["Average negative particle concentration [mol.m-3]"].entries /
            prev_params["Maximum concentration in negative electrode [mol.m-3]"] * Qn +
            sol["Average positive particle concentration [mol.m-3]"].entries /
            prev_params["Maximum concentration in positive electrode [mol.m-3]"] * Qp) * 3600 / F

    L_sei = sol["X-averaged negative SEI thickness [m]"].entries
    L_pl = sol["X-averaged negative dead lithium thickness [m]"].entries

    return {"n_Li": n_Li, "L_sei": L_sei, "L_pl": L_pl, "eps_n": eps_n, "eps_p": eps_p}


def extract_y0_yend(sol, prev_params):
    res = extract_y_from_solution(sol, prev_params)
    return {k: v[0] for k, v in res.items()}, {k: v[-1] for k, v in res.items()}


COARSE_KEYS = ["n_Li", "L_sei", "L_pl", "eps_n", "eps_p"]

def y_dict_to_vec(y):
    return np.array([y[k] for k in COARSE_KEYS], dtype=float)

def y_vec_to_dict(v):
    return {k: float(v[i]) for i, k in enumerate(COARSE_KEYS)}


def update_input_params(param_input, extrap_vars, prev_params):
    x_0, y_0, x_100, y_100 = extract_stoich_from_param(extrap_vars, prev_params)

    param_input["Initial concentration in negative electrode [mol.m-3]"] = (
        x_100 * prev_params["Maximum concentration in negative electrode [mol.m-3]"])
    param_input["Initial concentration in positive electrode [mol.m-3]"] = (
        y_100 * prev_params["Maximum concentration in positive electrode [mol.m-3]"])
    param_input["Initial SEI thickness [m]"] = extrap_vars["L_sei"]

    a_n = extrap_vars["eps_n"] * 3 / prev_params["Negative particle radius [m]"]
    V_Li = prev_params["Lithium metal partial molar volume [m3.mol-1]"]
    param_input["Initial plated lithium concentration [mol.m-3]"] = extrap_vars["L_pl"] * a_n / V_Li

    param_input["Negative electrode active material volume fraction"] = extrap_vars["eps_n"]
    param_input["Positive electrode active material volume fraction"] = extrap_vars["eps_p"]

    return param_input


def eval_delta_y(y_vec, sim, param_input, prev_params, parameter_values):
    y_dict = y_vec_to_dict(y_vec)
    param_input_sol = deepcopy(parameter_values)
    param_input_sol = update_input_params(param_input_sol, y_dict, prev_params)
    sol_fast = sim.solve(inputs=param_input_sol)

    _, yend_dict = extract_y0_yend(sol_fast.cycles[-1], prev_params)
    y_end_vec = y_dict_to_vec(yend_dict)

    return y_end_vec - y_vec, sol_fast, param_input_sol


def scipy_rk23_integrate(y0_vec, sim, param_input, prev_params, parameter_values,
                         t_final, max_step, rtol=1e-2, atol=1e-2,
                         stop_condition_fn=None):

    def rhs(t, y):
        dy, sol, _ = eval_delta_y(y, sim, rhs.param, prev_params, parameter_values)
        rhs.last_sol = sol
        return dy

    rhs.param = param_input
    rhs.last_sol = None

    solver = RK23(rhs, t0=0.0, y0=y0_vec, t_bound=t_final,
                  max_step=max_step, rtol=rtol, atol=atol)

    history = []
    idx = 1

    while solver.status == "running":
        start = time.time()
        solver.step()
        elapsed = time.time() - start

        t = solver.t
        y = solver.y.copy()
        sol = rhs.last_sol

        history.append((t, y, sol))
        print(f"Step {idx} at cycle {t:.1f} in {elapsed:.1f}s")
        idx += 1

        if y[3] < 0.1 or y[4] < 0.1:
            break

        if stop_condition_fn is not None and stop_condition_fn(t, y, sol):
            break

    return history


def get_capacity_from_sol(solution, prev_params, idx, esoh_solver):
    F = 96485.33
    eps_n = solution["X-averaged negative electrode active material volume fraction"].entries[idx]
    Qn = (eps_n * prev_params["Negative electrode thickness [m]"] *
          prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
          prev_params["Maximum concentration in negative electrode [mol.m-3]"] * F / 3600.)

    eps_p = solution["X-averaged positive electrode active material volume fraction"].entries[idx]
    Qp = (eps_p * prev_params["Positive electrode thickness [m]"] *
          prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
          prev_params["Maximum concentration in positive electrode [mol.m-3]"] * F / 3600.)

    Qp_0 = (solution["X-averaged positive particle concentration"].entries[-1][idx] *
            solution["Positive electrode capacity [A.h]"].entries[idx])
    Qn_0 = (solution["X-averaged negative particle concentration"].entries[-1][idx] *
            solution["Negative electrode capacity [A.h]"].entries[idx])
    QLi = Qp_0 + Qn_0

    inputs = {"V_min": 2.5, "V_max": 4.2, "Q_n": Qn, "Q_p": Qp, "Q_Li": QLi}
    esoh_sol = esoh_solver.solve(inputs)
    x_0 = esoh_sol["x_0"]
    x_100 = esoh_sol["x_100"]

    return eps_n * prev_params["Negative electrode thickness [m]"] * \
           prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] * \
           prev_params["Maximum concentration in negative electrode [mol.m-3]"] * F * (x_100 - x_0) / 3600


if __name__ == "__main__":

    tot_cycles = 10000
    initial_soc = 1.

    param_chen = pybamm.ParameterValues("Chen2020")
    param = pybamm.ParameterValues("OKane2022")
    param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
    param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
    param["Electrolyte diffusivity [m2.s-1]"] = param["Electrolyte diffusivity [m2.s-1]"](1., 298.15)

    param["EC diffusivity [m2.s-1]"] /= 150
    param["Positive electrode LAM constant proportional term [s-1]"] /= 2000.
    param["Negative electrode LAM constant proportional term [s-1]"] *= 20.
    param["Lithium plating kinetic rate constant [m.s-1]"] /= 100.
    param["Upper voltage cut-off [V]"] = 4.2

    param = update_LAM_params(param)
    param.set_initial_state(initial_soc)
    prev_params = deepcopy(param)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    model = pybamm.lithium_ion.SPMe(
        options={"SEI": "ec reaction limited", "x-average side reactions": "false"},
        build=False)

    model.submodels["Negative active material"] = StressBasedLAM_FP(
        model.param, "Negative",
        options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.submodels["Plating"] = Plating_FP(
        model.param, "negative", x_average=True,
        options={"lithium plating": "irreversible"})

    model.submodels["Positive active material"] = StressBasedLAM_FP(
        model.param, "Positive",
        options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.build_model()

    # Compute initial n_Li
    F = 96485.33
    eps_n0 = prev_params["Negative electrode active material volume fraction"]
    eps_p0 = prev_params["Positive electrode active material volume fraction"]
    Qn0 = (eps_n0 * prev_params["Negative electrode thickness [m]"] *
           prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
           prev_params["Maximum concentration in negative electrode [mol.m-3]"] * F / 3600.)
    Qp0 = (eps_p0 * prev_params["Positive electrode thickness [m]"] *
           prev_params["Electrode height [m]"] * prev_params["Electrode width [m]"] *
           prev_params["Maximum concentration in positive electrode [mol.m-3]"] * F / 3600.)
    c_n_ini = prev_params["Initial concentration in negative electrode [mol.m-3]"]
    c_p_ini = prev_params["Initial concentration in positive electrode [mol.m-3]"]
    n_Li0 = (c_n_ini / prev_params["Maximum concentration in negative electrode [mol.m-3]"] * Qn0 +
             c_p_ini / prev_params["Maximum concentration in positive electrode [mol.m-3]"] * Qp0) * 3600 / F

    # Load dynamic drive cycle
    cycle = np.loadtxt("../../../Dynamic_Profiles/WLTC.txt", skiprows=1)
    cycle[:, 0] -= 1
    cycle[:, 1] *= -1
    drive_step = pybamm.step.current(cycle, termination="<2.5V", duration=1800000)

    param_input = {
        "Initial concentration in negative electrode [mol.m-3]": prev_params["Initial concentration in negative electrode [mol.m-3]"],
        "Initial concentration in positive electrode [mol.m-3]": prev_params["Initial concentration in positive electrode [mol.m-3]"],
        "Initial SEI thickness [m]": prev_params["Initial SEI thickness [m]"],
        "Initial plated lithium concentration [mol.m-3]": prev_params["Initial plated lithium concentration [mol.m-3]"] * 0,
        "Negative electrode active material volume fraction": prev_params["Negative electrode active material volume fraction"],
        "Positive electrode active material volume fraction": prev_params["Positive electrode active material volume fraction"],
    }

    param.update({
        "Initial concentration in negative electrode [mol.m-3]": "[input]",
        "Initial concentration in positive electrode [mol.m-3]": "[input]",
        "Initial SEI thickness [m]": "[input]",
        "Initial plated lithium concentration [mol.m-3]": "[input]",
        "Negative electrode active material volume fraction": "[input]",
        "Positive electrode active material volume fraction": "[input]",
    })

    experiment = pybamm.Experiment([
        drive_step,
        "Charge at 0.3C until 4.2 V",
        "Hold at 4.2V until C/50",
    ])
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)

    y0_vec = y_dict_to_vec({
        "n_Li": n_Li0,
        "L_sei": prev_params["Initial SEI thickness [m]"],
        "L_pl": 0.,
        "eps_n": prev_params["Negative electrode active material volume fraction"],
        "eps_p": prev_params["Positive electrode active material volume fraction"],
    })

    start_time = time.time()
    history = scipy_rk23_integrate(
        y0_vec=y0_vec,
        sim=sim,
        param_input=param_input,
        prev_params=prev_params,
        parameter_values=param,
        t_final=float(tot_cycles),
        max_step=tot_cycles,
        rtol=1e-3,
        atol=1e-3,
    )
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")

    output_file = "simulation_history_WLTC_sulzer_v2.pkl"
    # Only (t, y) are ever read back (see dynamic_ours_v2.py); `sol` is a
    # pybamm.Solution holding compiled CasADi objects whose serialization
    # format is tied to the CasADi version, so it must not be pickled.
    history_to_save = [(t, y, None) for (t, y, _) in history]
    with open(output_file, "wb") as f:
        pickle.dump({"history": history_to_save, "prev_params": prev_params}, f)
    print(f"Saved to {output_file}")

    cycles = [t for (t, _, _) in history]
    all_Qs = [get_capacity_from_sol(s, prev_params, 0, esoh_solver) for (_, _, s) in history]
    Lsei = [y[1] * 1e9 for (_, y, _) in history]
    Lpl = [y[2] * 1e9 for (_, y, _) in history]
    eps_n = [y[3] for (_, y, _) in history]
    eps_p = [y[4] for (_, y, _) in history]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].scatter(cycles, all_Qs)
    axes[0, 0].set_xlabel("Cycle")
    axes[0, 0].set_ylabel("Q [Ah]")

    axes[0, 1].scatter(cycles, Lsei, label="SEI")
    axes[0, 1].scatter(cycles, Lpl, label="Plating")
    axes[0, 1].set_xlabel("Cycle")
    axes[0, 1].set_ylabel("Thickness [nm]")
    axes[0, 1].legend()

    axes[1, 0].scatter(cycles, eps_n)
    axes[1, 0].set_xlabel("Cycle")
    axes[1, 0].set_ylabel("eps_n")

    axes[1, 1].scatter(cycles, eps_p)
    axes[1, 1].set_xlabel("Cycle")
    axes[1, 1].set_ylabel("eps_p")

    plt.tight_layout()
    plt.show()
