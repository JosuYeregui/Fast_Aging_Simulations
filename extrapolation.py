import pybamm
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def extract_y0(sol, param):
    res = extract_y_from_solution(sol, param)
    return {k: v[0] for k, v in res.items()}

def extract_yend(sol, param):
    res = extract_y_from_solution(sol, param)
    return {k: v[-1] for k, v in res.items()}

def extract_y_from_solution(sol, param):

    Qp = sol["X-averaged positive particle concentration"].entries[-1] * \
         sol["Positive electrode capacity [A.h]"].entries
    Qn = sol["X-averaged negative particle concentration"].entries[-1] * \
         sol["Negative electrode capacity [A.h]"].entries
    nLi = 3600 / 96485.33 * (Qp + Qn)

    L_sei = sol["X-averaged negative SEI thickness [m]"].entries

    eps_n = param["Negative electrode active material volume fraction"] * (1 - sol["Loss of active material in negative electrode [%]"].entries / 100.)

    eps_p = param["Positive electrode active material volume fraction"] * (1 - sol["Loss of active material in positive electrode [%]"].entries / 100.)

    return {"nLi": nLi, "L_sei": L_sei, "eps_n": eps_n, "eps_p": eps_p}

if __name__ == "__main__":

    experiment = pybamm.Experiment(
        [(
                "Discharge at 1C until 3V",
                "Rest for 1 hour",
                "Charge at 1C until 4.2V",
                "Hold at 4.2V until C/50",)
        ] * 100
    )

    parameter_values = pybamm.ParameterValues("Mohtat2020")
    parameter_values.update({"SEI kinetic rate constant [m.s-1]": 1e-14})
    parameter_values.set_initial_stoichiometries(1)

    model = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited"})
    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)

    sol = sim.solve()

    print(sorted(sol.summary_variables.all_variables))
    Qp = sol["X-averaged positive particle concentration"].entries[-1][-1] * sol["Positive electrode capacity [A.h]"].entries[-1]
    Qn = sol["X-averaged negative particle concentration"].entries[-1][-1] * sol["Negative electrode capacity [A.h]"].entries[-1]
    nLi = 3600/96485.33*(Qp+Qn)
    print("nLi: ", nLi)
    L_num = sol["X-averaged negative SEI thickness [m]"].entries[-1]
    print("L_num: ", L_num)
    print("nLI param: ", sol["Total lithium [mol]"].entries[-1])
    print(sol["Negative SEI thickness [m]"].entries[-1])
    Qs = [s.steps[3]["Discharge capacity [A.h]"].entries[-1] for s in sol.cycles]
    fig, ax = plt.subplots()
    ax.plot(Qs)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Discharge capacity [A.h]")
    plt.show()
    print(max(sol["Discharge capacity [A.h]"].entries))
    print(parameter_values["Negative electrode active material volume fraction"], parameter_values["Positive electrode active material volume fraction"])

