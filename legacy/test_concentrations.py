import pybamm
from utils.pybamm_submodels import *
from copy import deepcopy
import time

initial_soc = 1.
tot_cycles = 1

param_chen = pybamm.ParameterValues("Chen2020")
param = pybamm.ParameterValues("OKane2022")
param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
param["Electrolyte diffusivity [m2.s-1]"] = param["Electrolyte diffusivity [m2.s-1]"](1., 298.15)
# param["Ratio of lithium moles to SEI moles"] = 2.

param["EC diffusivity [m2.s-1]"] /= 150
param["Positive electrode LAM constant proportional term [s-1]"] /= 2000.
param["Negative electrode LAM constant proportional term [s-1]"] *= 20.
param["Lithium plating kinetic rate constant [m.s-1]"] /= 100.

param["Upper voltage cut-off [V]"] = 4.2

param = update_LAM_params(param)

param.set_initial_state(initial_soc)

prev_params = deepcopy(param)

esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

# spm = pybamm.lithium_ion.SPM({"SEI": "ec reaction limited", "loss of active material": "stress-driven","x-average side reactions":"true"})

model = pybamm.lithium_ion.SPMe(
    options={"SEI": "ec reaction limited", "x-average side reactions": "false"},
    build=False)

# model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={
#     "loss of active material": "stress-driven"}, x_average=True, phase="primary")

#model.submodels["Plating"] = Plating_FP(model.param, "negative", x_average=True, options={"lithium plating": "irreversible"})

# model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={
#     "loss of active material": "stress-driven"}, x_average=True, phase="primary")

model.build_model()

solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4)

experiment = pybamm.Experiment(["Discharge at 0.5C until 2.5 V", "Rest for 1 hour", "Charge at 0.3C until 4.2 V","Hold at 4.2V until C/50"]*100)
start_time = time.time()
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param, solver=solver)
base_sol = sim.solve(initial_soc=initial_soc)
print("Baseline time: ", time.time() - start_time)

model_clean = pybamm.lithium_ion.SPMe()
sim_clean = pybamm.Simulation(model_clean, experiment=experiment, parameter_values=prev_params, solver=solver)
start_time = time.time()
clean_sol = sim_clean.solve(initial_soc=initial_soc)
print("Clean time: ", time.time() - start_time)

Qp_0_bs = base_sol["X-averaged positive particle concentration"].entries[-1][-1] * \
       base_sol["Positive electrode capacity [A.h]"].entries[-1]
Qn_0_bs = base_sol["X-averaged negative particle concentration"].entries[-1][-1] * \
       base_sol["Negative electrode capacity [A.h]"].entries[-1]
QLi_bs = (Qp_0_bs + Qn_0_bs)

Qp_0_cl = clean_sol["X-averaged positive particle concentration"].entries[-1][-1] * \
       clean_sol["Positive electrode capacity [A.h]"].entries[-1]
Qn_0_cl = clean_sol["X-averaged negative particle concentration"].entries[-1][-1] * \
       clean_sol["Negative electrode capacity [A.h]"].entries[-1]
QLi_cl = (Qp_0_cl + Qn_0_cl)

lost = base_sol["Loss of capacity to negative SEI [A.h]"].entries[-1] + base_sol["Loss of capacity to negative lithium plating [A.h]"].entries[-1]

print("QLi_bs: ", QLi_bs)
print("QLi_cl: ", QLi_cl)

print("QLi_cl - lost", QLi_cl - lost)

