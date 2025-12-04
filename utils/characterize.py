import pybamm
import numpy as np

def get_Qn(param, eps_n):

    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    return eps_n * L * cs_max * F * A / 3600

def get_Q(param):

    x_0, x_100, y_100, y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(param)

    eps_n = param["Negative electrode active material volume fraction"]
    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    return eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600

def OCV_func_param(soc, param):
    x_0, x_100, y_100, y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(param)

    sol_p = soc * (y_100 - y_0) + y_0
    OCP_p = param["Positive electrode OCP [V]"](sol_p)

    sol_n = soc * (x_100 - x_0) + x_0
    OCP_n = param["Negative electrode OCP [V]"](sol_n)

    return OCP_p - OCP_n

def get_OVC_func(param):
    return lambda soc: OCV_func_param(soc, param)

def get_Rint(model, param):
    experiment = pybamm.Experiment(["Charge at 1C until 4.2V", "Discharge at 1C until 2.5V"])
    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=param
    )

    solution = sim.solve(initial_soc=0.)

    V = solution["Terminal voltage [V]"].data
    OCV = solution["Battery open-circuit voltage [V]"].data
    I = -solution["Current [A]"].data

    zero_current_mask = np.abs(I) < 1e-6

    V_pol = V - OCV

    R_int = np.full_like(I, np.nan)

    non_zero_indices = ~zero_current_mask
    R_int[non_zero_indices] = V_pol[non_zero_indices] / I[non_zero_indices]
    median_R_int = np.median(R_int)

    return np.abs(median_R_int)


if __name__ == "__main__":
    param = pybamm.ParameterValues("Chen2020")
    model = pybamm.lithium_ion.SPMe()

    print(get_Rint(model, param))

    print(pybamm.lithium_ion.get_min_max_stoichiometries(param))

    print(get_Q(param))