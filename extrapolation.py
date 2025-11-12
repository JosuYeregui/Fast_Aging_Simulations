import pybamm
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def extract_stoich_from_sol(solution, param, idx):
    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    Qn = solution["Negative electrode capacity [A.h]"].entries[idx]
    Qp = solution["Positive electrode capacity [A.h]"].entries[idx]
    Qp_0 = solution["X-averaged positive particle concentration"].entries[-1][idx] * \
           solution["Positive electrode capacity [A.h]"].entries[idx]
    Qn_0 = solution["X-averaged negative particle concentration"].entries[-1][idx] * \
           solution["Negative electrode capacity [A.h]"].entries[idx]
    QLi = (Qp_0 + Qn_0)

    inputs = {"V_min": 2.8, "V_max": 4.2,
              "Q_n": Qn,
              "Q_p": Qp,
              "Q_Li": QLi}

    esoh_sol = esoh_solver.solve(inputs)

    x_0 = esoh_sol["x_0"]
    y_0 = esoh_sol["y_0"]
    x_100 = esoh_sol["x_100"]
    y_100 = esoh_sol["y_100"]

    return x_0, y_0, x_100, y_100

def extract_y0_yend(sol, param):
    res = extract_y_from_solution(sol, param)
    return {k: v[0] for k, v in res.items()}, {k: v[-1] for k, v in res.items()}

def extract_y_from_solution(sol, param):

    Qp = sol["X-averaged positive particle concentration"].entries[-1] * \
         sol["Positive electrode capacity [A.h]"].entries
    Qn = sol["X-averaged negative particle concentration"].entries[-1] * \
         sol["Negative electrode capacity [A.h]"].entries
    nLi = 3600 / 96485.33 * (Qp + Qn)
    L_sei = sol["X-averaged negative SEI thickness [m]"].entries
    eps_n = sol["X-averaged negative electrode active material volume fraction"].entries
    eps_p = sol["X-averaged positive electrode active material volume fraction"].entries
    # eps_n = param["Negative electrode active material volume fraction"] * (1 - sol["Loss of active material in negative electrode [%]"].entries / 100.)
    # eps_p = param["Positive electrode active material volume fraction"] * (1 - sol["Loss of active material in positive electrode [%]"].entries / 100.)

    return {"nLi": nLi, "L_sei": L_sei, "eps_n": eps_n, "eps_p": eps_p}

def extrapolate_solution(solution, param, n_cycles):

    vars_0, vars_end = extract_y0_yend(solution, param)

    extrapolated_vars = {}

    for k_0, v_0 in vars_0.items():
        v_end = vars_end[k_0]
        extrapolated_vars[k_0] = v_0 + (v_end - v_0) / (n_cycles - 1) * n_cycles

    return extrapolated_vars

if __name__ == "__main__":

    experiment_baseline = pybamm.Experiment(
        [(
                "Discharge at 1C until 3V",
                "Rest for 1 hour",
                "Charge at 1C until 4.2V",
                "Hold at 4.2V until C/50",)
        ] * 1000
    )

    parameter_values = pybamm.ParameterValues("Mohtat2020")
    parameter_values.update({"SEI kinetic rate constant [m.s-1]": 1e-15})
    parameter_values.update({"Negative electrode LAM constant proportional term [s-1]": 1.7e-7}, check_already_exists=False)  # Sulzer 2021 std
    parameter_values.update({"Positive electrode LAM constant proportional term [s-1]": 1.7e-7}, check_already_exists=False)

    param_Ai = pybamm.ParameterValues("Ai2020")
    parameters_to_change = ["Negative electrode partial molar volume [m3.mol-1]",
                            "Positive electrode partial molar volume [m3.mol-1]",
                            "Negative electrode LAM constant exponential term",
                            "Positive electrode LAM constant exponential term",
                            "Negative electrode LAM constant proportional term [s-1]",
                            "Positive electrode LAM constant proportional term [s-1]",
                            "Negative electrode critical stress [Pa]",
                            "Positive electrode critical stress [Pa]",
                            "Negative electrode Young's modulus [Pa]",
                            "Positive electrode Young's modulus [Pa]",
                            "Negative electrode Poisson's ratio",
                            "Positive electrode Poisson's ratio",
                            "Negative electrode reference concentration for free of deformation [mol.m-3]",
                            "Positive electrode reference concentration for free of deformation [mol.m-3]",
                            "Negative electrode volume change",
                            "Positive electrode volume change",
                            "Cell thermal expansion coefficient [m.K-1]"]
    for p in parameters_to_change:
        parameter_values.update({p: param_Ai[p]}, check_already_exists=False)

    #parameter_values.update({"Negative electrode partial molar volume [m3.mol-1]": param_Ai[]}, check_already_exists=False)
    #parameter_values.update({"Positive electrode partial molar volume [m3.mol-1]": 0.}, check_already_exists=False)

    parameter_values.set_initial_stoichiometries(1)

    spm = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited", "loss of active material": "stress-driven", "particle mechanics": "swelling only", "stress-induced diffusion": "true"})
    sim_baseline = pybamm.Simulation(spm, experiment=experiment_baseline, parameter_values=parameter_values)

    sol_baseline = sim_baseline.solve()

    experiment = pybamm.Experiment(
        [(
            "Discharge at 1C until 3V",
            "Rest for 1 hour",
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/50",)
        ]
    )

    starting_sol = None
    for i in range(100):

        sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=parameter_values)
        if starting_sol is not None:
            sol = sim.solve(starting_solution=starting_sol)
        else:
            sol = sim.solve()
        # Last cycle containing the steps of the defined experiment sequence
        sol_last = sol.cycles[-1]
        # Extrapolate variables using the last cycle simulation information
        extrapolated_vars = extrapolate_solution(sol_last, parameter_values, 10)

        # Hemen parameter load-a egin behar dugu

        starting_sol = sol_last

    # print(extract_stoich_from_sol(sol, parameter_values, 0))
    # print(extract_stoich_from_sol(sol, parameter_values, -1))
    #
    # print(sorted(sol.summary_variables.all_variables))
    # Qp = sol["X-averaged positive particle concentration"].entries[-1][-1] * sol["Positive electrode capacity [A.h]"].entries[-1]
    # Qn = sol["X-averaged negative particle concentration"].entries[-1][-1] * sol["Negative electrode capacity [A.h]"].entries[-1]
    # nLi = 3600/96485.33*(Qp+Qn)
    # print("nLi: ", nLi)
    # L_num = sol["X-averaged negative SEI thickness [m]"].entries[-1]
    # print("L_num: ", L_num)
    # print("nLI param: ", sol["Total lithium [mol]"].entries[-1])
    # print(sol["Negative SEI thickness [m]"].entries[-1])
    # Qs = [s.steps[1]["Discharge capacity [A.h]"].entries[-1] for s in sol.cycles]
    # fig, ax = plt.subplots()
    # ax.plot(Qs)
    # ax.set_xlabel("Cycle")
    # ax.set_ylabel("Discharge capacity [A.h]")
    # plt.show()
    # print(max(sol["Discharge capacity [A.h]"].entries))
    # print(parameter_values["Negative electrode active material volume fraction"], parameter_values["Positive electrode active material volume fraction"])
    # print(sol["X-averaged negative electrode active material volume fraction"].entries[-1], sol["X-averaged positive electrode active material volume fraction"].entries[-1])

