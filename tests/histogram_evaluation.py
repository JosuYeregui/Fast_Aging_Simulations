import numpy as np
import matplotlib.pyplot as plt
import pybamm

from main import scipy_rk23_integrate
from utils.param_to_inputvars import load_params
from utils.cycler import CCStep, CCCVStep, Cycler, DynamicProfile
from utils.characterize import *
from utils.pybamm_submodels import *

def get_ce_cs(I, input_vars):


    ce_int = (input_vars["ce_0"] + I * (1 - input_vars["t_plus"]) / input_vars["F"] *
              (input_vars["L_n"] / (3 * input_vars["Deff_n"])))

    # ce_int = np.maximum(ce_int, 1e-10)
    cs_cave = input_vars["cs_cave_n"](-I * input_vars["A"], input_vars["eps_0_n"])

    return ce_int, cs_cave

def get_ce_cs_pybamm(I):
    model = pybamm.lithium_ion.SPMe(
        options={"SEI": "ec reaction limited", "x-average side reactions": "false", "lithium plating": "irreversible"},
        build=False)

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

    model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")

    # model.submodels["Positive active material"] = StressBasedLAM(
    #     model.param, "Positive"
    # )

    model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.build_model()

    if I > 0:
        exp = f"Charge at {I} A for 100 hours or until 4.2 V"
        soc0 = 0.
    elif I < 0:
        exp = f"Discharge at {abs(I)} A for 100 hours or until 2.5 V"
        soc0 = 1.
    else:
        exp = "Rest for 1 hour"
        soc0 = 0.5

    experiment = pybamm.Experiment([
        exp,
    ], )

    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    sim.solve(initial_soc=soc0)
    sol = sim.solution
    t = sol["Time [s]"].entries

    cs_cave = sol["X-averaged negative particle surface concentration [mol.m-3]"](t) - sol[
        "Average negative particle concentration [mol.m-3]"](t)

    ce = sol["X-averaged negative electrolyte concentration [mol.m-3]"](t)

    return ce, cs_cave, t

num_nodes = 1000
tolerance = 0.01
times_to_ss = []
steady_state_errors = []
I_app = []

input_vars = load_params("../input/M50_okane_modified_plating.yaml")
input_vars["D_ec"] /= 20
input_vars["Beta_p"] /= 600
input_vars["Beta_n"] *= 20

for i in range(num_nodes):

    I1 = np.random.uniform(-10, -0.1)
    I2 = np.random.uniform(0.1, 10)
    I = np.random.choice([I1, I2])
    I_app.append(I)

    ce, cs_cave, t = get_ce_cs_pybamm(I)
    ce_anal, cs_cave_anal = get_ce_cs(-I/input_vars["A"], input_vars)

    rel_diff = np.abs((cs_cave - cs_cave_anal) / cs_cave_anal)

    # Find indices where the curve is within the 1% tolerance limit
    reached_ss_indices = np.where(rel_diff <= tolerance)[0]

    if len(reached_ss_indices) > 0:
        first_idx = reached_ss_indices[0]
        times_to_ss.append(t[first_idx])
    else:
        # If it never reaches the tolerance limit, append NaN
        times_to_ss.append(np.nan)
        first_idx = 0

    steady_state_errors.append(np.sqrt(np.mean(((cs_cave[first_idx:] - cs_cave_anal)/cs_cave_anal) ** 2)))

I_app_t = np.abs(np.array(I_app))
times_to_ss = np.array(times_to_ss)
nonvalid_indices = np.isnan(times_to_ss) | (times_to_ss > 3600.)
times_to_ss = times_to_ss[~nonvalid_indices]
I_app_t = I_app_t[~nonvalid_indices]
steady_state_errors = np.array(steady_state_errors)[~np.isnan(steady_state_errors)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Steady-State Error Histogram
ax1.hist(steady_state_errors, bins=15, color='#3498db', edgecolor='black', alpha=0.8)
ax1.set_title('Error in Steady State\n(|PyBaMM Final - Analytical|)')
ax1.set_xlabel('Relative Error')
ax1.set_ylabel('Frequency (Number of Nodes)')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Time to Steady State Histogram
# Filter out any NaNs just in case some nodes didn't converge
valid_times = times_to_ss[~np.isnan(times_to_ss)]
ax2.scatter(I_app_t, valid_times, color='#e74c3c', alpha=0.6)
m, b = np.polyfit(I_app_t, valid_times, 1)
ax2.plot(I_app_t, m * I_app_t + b, color='black', linestyle='--')
ax2.set_title(f'Time to Reach Steady State\n(Within {tolerance*100}% tolerance)')
ax2.set_xlabel('|Current [A]|')
ax2.set_ylabel('Time to Reach Steady State [s]')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

