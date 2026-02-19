import numpy as np
import matplotlib.pyplot as plt
import pybamm

from main import scipy_rk23_integrate
from utils.param_to_inputvars import load_params
from utils.cycler import CCStep, CCCVStep, Cycler, DynamicProfile
from utils.characterize import *
from utils.pybamm_submodels import *

def dPlating_dt(t, y, cycler, input_vars):
    I_raw, soc = cycler(t/3600.)
    I_raw = -I_raw

    A = input_vars["A"]
    F = input_vars["F"]
    R = input_vars["R"]
    T = input_vars["T"]
    RTF = R * T / F

    I = I_raw / A

    as_n = 3 * y[1] / input_vars["R_n"]  # specific surface area of the negative electrode

    ce_int_s = (input_vars["ce_0"] + I * (1 - input_vars["t_plus"]) / F *
              (input_vars["L_n"] / (3 * input_vars["Deff_n"]) - input_vars["L_s"] / (2 * input_vars["Deff_s"])))
    ce_int = (input_vars["ce_0"] + I * (1 - input_vars["t_plus"]) / F *
              (input_vars["L_n"] / (3 * input_vars["Deff_n"])))
    ce_int =  ce_int

    sol_n = soc * (input_vars["x_100"] - input_vars["x_0"]) + input_vars["x_0"]
    sol_n += input_vars["cs_cave_n"](I_raw, y[1]) / input_vars["csmax_n"]  # We need surface concentration

    j0 = np.sqrt(ce_int) * np.sqrt(sol_n * input_vars["csmax_n"]) * np.sqrt(
        input_vars["csmax_n"] - sol_n * input_vars["csmax_n"]) * input_vars["m_ref"]
    # j0 = np.maximum(j0, 1e-10)  # Avoid division by zero in asinh

    eta_r = -2 * RTF * np.arcsinh(I / (as_n * j0 * input_vars["L_n"]))

    ocp = input_vars["U_n"](sol_n)
    eta_plating = -(eta_r - ocp)
    i0 = input_vars["k_plating"] * ce_int * F
    # i0 = 1e-4*ce_int
    j_k = i0 * np.exp(-input_vars["alpha_plating"] * eta_plating / RTF)
    return j_k, ce_int, eta_plating

# --------------------------------------------------------------------------------
# 1. Model setup
# --------------------------------------------------------------------------------

model = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false", "lithium plating": "irreversible"}, build=False)

param_chen = pybamm.ParameterValues("Chen2020")
param = pybamm.ParameterValues("OKane2022")
param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
param["Electrolyte diffusivity [m2.s-1]"] = param["Electrolyte diffusivity [m2.s-1]"](1., 298.15)
# param["Ratio of lithium moles to SEI moles"] = 2.

param["EC diffusivity [m2.s-1]"] /= 20
param["Positive electrode LAM constant proportional term [s-1]"] /= 600.
param["Negative electrode LAM constant proportional term [s-1]"] *= 20.
param["Upper voltage cut-off [V]"] = 4.2

param = update_LAM_params(param)

# model.submodels["Negative active material"] = StressBasedLAM(
#     model.param, "Negative"
# )

model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

# model.submodels["Positive active material"] = StressBasedLAM(
#     model.param, "Positive"
# )

model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

model.build_model()

experiment = pybamm.Experiment([
    "Discharge at 0.2C until 2.5 V",
    "Charge at 0.2C until 4.2 V",
    "Hold at 4.2 V until C/10"
] * 10, period="1 second")

sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
print("Running simulation...")
sim.solve(initial_soc=1.)
sol = sim.solution
t = sol["Time [s]"].entries

# Extract plating variables
eta_key = "X-averaged negative electrode lithium plating reaction overpotential [V]"
j_key = "Negative electrode lithium plating interfacial current density [A.m-2]"
delta_phi = "X-averaged negative electrode surface potential difference [V]"
eta_t = sol[eta_key](t)
j_pl_num = sol["X-averaged negative electrode lithium plating interfacial current density [A.m-2]"](t)

# --------------------------------------------------------------------------------
# 2. Analytical parameters
# --------------------------------------------------------------------------------
F = 96485.33212
R = 8.314462618
T = sol["Ambient temperature [K]"](t)
V_Li = param["Lithium metal partial molar volume [m3.mol-1]"]
alpha_c = param["Lithium plating transfer coefficient"]
a = param["Negative electrode active material volume fraction"] * 3 / param["Negative particle radius [m]"]

c_Li = sol["X-averaged negative dead lithium concentration [mol.m-3]"](t)
c_e = sol["X-averaged negative electrolyte concentration [mol.m-3]"](t)
k_plating = param["Lithium plating kinetic rate constant [m.s-1]"]
i0 = k_plating*F*c_e

# Numerical PyBaMM result for comparison
L_num = V_Li / a * sol["X-averaged negative dead lithium concentration [mol.m-3]"](t)
L_num_dead = sol["X-averaged negative dead lithium thickness [m]"](t) #Gorakuan berdina
eta_pl_pb = sol["X-averaged negative electrode lithium plating reaction overpotential [V]"](t)
j_pl_pb = sol["X-averaged negative electrode lithium plating interfacial current density [A.m-2]"](t)
LLI = sol["Loss of lithium inventory [%]"](t)

initial_soc = 1.


# -------------- Load parameters ----------------

input_vars = load_params("../input/M50_okane_modified_plating.yaml")
input_vars["D_ec"] /= 20
input_vars["Beta_p"] /= 600
input_vars["Beta_n"] *= 20


# ---------- Define simulation steps -------------

steps = [
    CCStep(I=-1., V_end=2.5),
    CCCVStep(I=1., V_end=4.2, Cutoff_I=0.5, Vlim=[2.5, 4.2]),
]

# --------------- Define cycler ------------------

OCV_fn = get_OVC_func(input_vars)

cycl = Cycler(
    steps=steps,
    SOC0=initial_soc,
    R=input_vars["R_int"],
    Q=input_vars["Q"],
    OCV_fn=OCV_fn
)

input_vars = set_c_ini(cycl, input_vars)

# -------------- Run simulation ---------------

y0_vec = [input_vars["L_0"], input_vars["eps_0_n"], input_vars["eps_0_p"], 0.]  # L, eps_n, eps_p, Plating

ce_guria = []
eta_pl_guria = []
j_guria = []
for t_i in t:
    j_pl_step, ce_step, eta_pl = dPlating_dt(t_i, y0_vec, cycl, input_vars)
    ce_guria.append(ce_step)
    eta_pl_guria.append(eta_pl)
    j_guria.append(-j_pl_step)

ce_pybamm = sol["X-averaged negative electrolyte concentration [mol.m-3]"](t)

fig, ax = plt.subplots()
ax.plot(t/3600, ce_guria, label="Guria")
ax.plot(t/3600, ce_pybamm, label="PyBaMM")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(t/3600, eta_pl_guria, label="Guria")
ax.plot(t/3600, eta_pl_pb, label="PyBaMM")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(t/3600, j_guria, label="Guria")
ax.plot(t/3600, j_pl_pb, label="PyBaMM")
ax.legend()
plt.show()