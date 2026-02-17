import pybamm
import numpy as np
import time

def set_c_ini(cycler, input_vars):
    soc_ave = cycler.get_soc_ave()
    sol_n = soc_ave * (input_vars["x_100"] - input_vars["x_0"]) + input_vars["x_0"]
    input_vars["cini_n_0"] = sol_n * input_vars["csmax_n"]
    sol_p = soc_ave * (input_vars["y_100"] - input_vars["y_0"]) + input_vars["y_0"]
    input_vars["cini_p_0"] = sol_p * input_vars["csmax_p"]

    return input_vars


def get_Qn(input_vars, eps_n):

    L = input_vars["L_n"]
    A = input_vars["A"]
    cs_max = input_vars["csmax_n"]
    F = 96485.33

    return eps_n * L * cs_max * F * A / 3600

def get_Qp(input_vars, eps_p):

    L = input_vars["L_p"]
    A = input_vars["A"]
    cs_max = input_vars["csmax_p"]
    F = 96485.33

    return eps_p * L * cs_max * F * A / 3600

def get_Q(input_vars, y):
    # x_0, x_100, y_100, y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(param)
    x_0 = input_vars["x_0"]
    x_100 = input_vars["x_100"]

    eps_n = y[1]
    L = input_vars["L_n"]
    A = input_vars["A"]
    cs_max = input_vars["csmax_n"]
    F = 96485.33

    return eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600

def stoichiometric_window(input_vars, y):
    
    eps_n = y[1]
    Qn = get_Qn(input_vars, eps_n)
    eps_p = y[2]
    Qp = get_Qp(input_vars, eps_p)

    csmax_n = input_vars["csmax_n"]
    theta_n = input_vars["cini_n"] / csmax_n
    OCP_n = input_vars["U_n"]

    csmax_p = input_vars["csmax_p"]
    theta_p = input_vars["cini_p"] / csmax_p
    OCP_p = input_vars["U_p"]
    
    
    Q_at_Vmin = find_voltage_limit(input_vars["V_min"], -input_vars["Q_nom"] * 2., input_vars["Q_nom"] * 0.5,
                                   theta_n, theta_p, Qn, Qp, OCP_n, OCP_p)
    Q_at_Vmax = find_voltage_limit(input_vars["V_max"], -input_vars["Q_nom"] * 0.5, input_vars["Q_nom"] * 2.,
                                   theta_n, theta_p, Qn, Qp, OCP_n, OCP_p)
    
    x_0 = theta_n + Q_at_Vmin / Qn
    x_100 = theta_n + Q_at_Vmax / Qn
    y_0 = theta_p - Q_at_Vmin / Qp
    y_100 = theta_p - Q_at_Vmax / Qp
    
    
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


def OCV_func_param(soc, input_vars):
    x_0 = input_vars["x_0"]
    x_100 = input_vars["x_100"]
    y_0 = input_vars["y_0"]
    y_100 = input_vars["y_100"]

    sol_p = soc * (y_100 - y_0) + y_0
    OCP_p = input_vars["U_p"](sol_p)

    sol_n = soc * (x_100 - x_0) + x_0
    OCP_n = input_vars["U_n"](sol_n)

    return OCP_p - OCP_n

def get_OVC_func(input_vars):
    return lambda soc: OCV_func_param(soc, input_vars)

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
    a_s = y[1] * 3 / input_vars["R_n"]
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

    return Q


def updated_model_R(y, input_vars, L0, R0):

    a_s =  y[1] * 3 / input_vars["R_n"]
    F = 96485.33

    R_dif = (y[0] - L0) * input_vars["rho_SEI"] / a_s * F
    R = R0 + R_dif

    return R


if __name__ == "__main__":
    param = pybamm.ParameterValues("Chen2020")
    model = pybamm.lithium_ion.SPMe()

    print(get_Rint(model, param))

    print(pybamm.lithium_ion.get_min_max_stoichiometries(param))
