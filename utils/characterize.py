import pybamm
import numpy as np
import time

def get_Qn(param, eps_n):

    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    return eps_n * L * cs_max * F * A / 3600

def get_Q(param):
    # x_0, x_100, y_100, y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(param)
    x_0, x_100, y_100, y_0 = stoichiometric_window(param)

    eps_n = param["Negative electrode active material volume fraction"]
    L = param["Negative electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    cs_max = param["Maximum concentration in negative electrode [mol.m-3]"]
    F = 96485.33

    return eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600

def stoichiometric_window(param):
    
    eps_n = param["Negative electrode active material volume fraction"]
    Qn = get_Qn(param, eps_n)
    eps_p = param["Positive electrode active material volume fraction"]
    csmax_p = param["Maximum concentration in positive electrode [mol.m-3]"]
    csmax_n = param["Maximum concentration in negative electrode [mol.m-3]"]
    L = param["Positive electrode thickness [m]"]
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    F = 96485.33
    Qp = eps_p * L * csmax_p * F * A / 3600
    theta_n = param["Initial concentration in negative electrode [mol.m-3]"] / csmax_n
    theta_p = param["Initial concentration in positive electrode [mol.m-3]"] / csmax_p
    OCP_p = param["Positive electrode OCP [V]"]
    OCP_n = param["Negative electrode OCP [V]"]
    
    
    Q_at_Vmin = find_voltage_limit(2.5, -8, 2, theta_n, theta_p, Qn, Qp, OCP_n, OCP_p)
    Q_at_Vmax = find_voltage_limit(4.2, -2, 8, theta_n, theta_p, Qn, Qp, OCP_n, OCP_p)
    
    x_0 = theta_n + Q_at_Vmin / Qn
    x_100 = theta_n + Q_at_Vmax / Qn
    y_0 = theta_p + Q_at_Vmin / Qp
    y_100 = theta_p + Q_at_Vmax / Qp
    
    
    return x_0, x_100, y_100, y_0

def bisection_for_voltage(
    voltage_fn,
    target_voltage,
    a,
    b,
    tol=1e-4,
    max_iter=100
):

    fa = voltage_fn(a) - target_voltage
    fb = voltage_fn(b) - target_voltage

    if fa * fb > 0:
        raise ValueError("Voltage does not cross target in the interval.")

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fmid = voltage_fn(mid) - target_voltage

        if abs(fmid) < tol:
            return mid

        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

    return mid

def cell_voltage_from_Q(Q, x_n0, x_p0, Cn, Cp, Un, Up):

    x_n = x_n0 + Q / Cn
    x_p = x_p0 - Q / Cp

    return Up(x_p) - Un(x_n)

def find_voltage_limit(
    target_voltage,
    Qmin,
    Qmax,
    x_n0, x_p0,
    Cn, Cp,
    Un, Up
):

    voltage_fn = lambda Q: cell_voltage_from_Q(Q, x_n0, x_p0, Cn, Cp, Un, Up)

    return bisection_for_voltage(voltage_fn, target_voltage, Qmin, Qmax)


def OCV_func_param(soc, param, stoich):
    x_0 = stoich["x_0"]
    x_100 = stoich["x_100"]
    y_0 = stoich["y_0"]
    y_100 = stoich["y_100"]

    sol_p = soc * (y_100 - y_0) + y_0
    OCP_p = param["Positive electrode OCP [V]"](sol_p)

    sol_n = soc * (x_100 - x_0) + x_0
    OCP_n = param["Negative electrode OCP [V]"](sol_n)

    return OCP_p - OCP_n

def get_OVC_func(param, stoich):
    return lambda soc: OCV_func_param(soc, param, stoich)

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