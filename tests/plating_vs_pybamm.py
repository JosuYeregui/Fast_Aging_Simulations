import numpy as np
import matplotlib.pyplot as plt
import pybamm

from main import scipy_rk23_integrate
from utils.param_to_inputvars import load_params
from utils.cycler import CCStep, CCCVStep, Cycler, DynamicProfile
from utils.characterize import *
from utils.pybamm_submodels import *

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
    "Discharge at 1C until 2.5 V",
    "Charge at 1C until 4.2 V",
    "Hold at 4.2 V until C/10"
] * 100, period="1 second")

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
L_num_dead = sol["X-averaged negative dead lithium thickness [m]"](t) #Gorakuan berdina
L_num_dead = sol["X-averaged negative dead lithium thickness [m]"](t)
LLI = sol["Loss of lithium inventory [%]"](t)

tot_cycles = 100
initial_soc = 1.


# -------------- Load parameters ----------------

input_vars = load_params("../input/M50_okane_modified_plating.yaml")
input_vars["D_ec"] /= 20
input_vars["Beta_p"] /= 600
input_vars["Beta_n"] *= 20


# ---------- Define simulation steps -------------

steps = [
    CCStep(I=-5., V_end=2.5),
    CCCVStep(I=5., V_end=4.2, Cutoff_I=0.5, Vlim=[2.5, 4.2]),
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

start_ode_time = time.time()
history = scipy_rk23_integrate(
    y0_vec=y0_vec,
    input_vars=input_vars,
    tot_cycles=tot_cycles,
    cycler=cycl,
    stop_Q=0.1,
    rtol=[1e-3, 1e-3, 1e-3, 1e-3],
    atol=[1e-9 ,1e-4, 1e-4, 1e-20],
    stop_condition_fn=None,  # optional early-stop
    max_step=None
)
ode_time = time.time() - start_ode_time
print("ODE + integration time: ", ode_time)

# -------------- Plotting ---------------
t_ode = history["t [s]"]
L_pl = history["y"][:, 3]

#t = t/t[-1] * 10.

fig, ax = plt.subplots()
# ax.plot(t, L_num, label="PyBaMM Plating Thickness")
ax.plot(t, L_num_dead, label="PyBaMM Dead Lithium Thickness")
ax.plot(t_ode, L_pl, label="ODE Plating Thickness", linestyle="--")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Plating Thickness (m)")
ax.set_title("Lithium Plating Thickness: PyBaMM vs ODE")
ax.legend()
ax.grid()
plt.show()