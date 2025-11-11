# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 10:42:03 2025

@author: ilopetegui
"""


import numpy as np
import matplotlib.pyplot as plt
import pybamm
import time

# --------------------------------------------------------------------------------
# 1. Build PyBaMM SPMe model with interstitial-diffusion-limited SEI
# --------------------------------------------------------------------------------
print("Building SPMe with interstitial-diffusion limited SEI option...")
model = pybamm.lithium_ion.SPMe(options={"loss of active material": "stress-driven","x-average side reactions":"false"})
param = pybamm.ParameterValues("Ai2020")

# --------------------------------------------------------------------------------
# 2. Extract parameters used by PyBaMM
# --------------------------------------------------------------------------------

j0 = float(param.get("SEI reaction exchange current density [A.m-2]",1.5e-7))      # or "SEI reaction exchange current density"
k_sei = float(param.get("SEI kinetic rate constant [m.s-1]",1e-12))
D_ec = float(param.get("EC diffusivity [m2.s-1]",2e-18))              # or "Molar mass of SEI"
c_ec = float(param.get("EC initial concentration in electrolyte [mol.m-3]",4541))
alpha = float(param.get("SEI charge transfer coefficient", 0.5))
U_sei = float(param.get("SEI open-circuit potential [V]",0.4))
F = float(param.get("Faraday constant [C.mol-1]", 96485.33212))
R = float(param.get("Ideal gas constant [J.mol-1.K-1]", 8.314462618))
T = float(param.get("Ambient temperature [K]", 298.15))
L0 = float(param.get("initial sei thickness",5e-9))
V_sei = float(param.get("sei partial molar volume",9.585e-5))
param.update({"Negative electrode LAM constant proportional term [s-1]": 1.7e-7}) #Sulzer 2021 std
param.update({"Positive electrode LAM constant proportional term [s-1]": 1.7e-7})

m_LAM_n = param["Negative electrode LAM constant exponential term"]
m_LAM_p = param["Positive electrode LAM constant exponential term"]
Beta_n = param["Negative electrode LAM constant proportional term [s-1]"]
Beta_p = param["Positive electrode LAM constant proportional term [s-1]"]
stress_critical_n = param["Negative electrode critical stress [Pa]"]
stress_critical_p = param["Positive electrode critical stress [Pa]"]




# --------------------------------------------------------------------------------
# 3. Build experiment: 50 charge/discharge cycles
# --------------------------------------------------------------------------------
n_cycles = 10
steps = []
for _ in range(n_cycles):
    steps.append("Charge at 1 C until 4.2 V")
    steps.append("Hold at 4.2 V until 0.02 C")
    steps.append("Discharge at 1 C until 2.5 V")
    # steps.append("Rest for 10 h")
experiment = pybamm.Experiment(steps, period="1 second")

# --------------------------------------------------------------------------------
# 4. Run PyBaMM simulation
# --------------------------------------------------------------------------------
start_time_pbm = time.time()
print(f"\nRunning PyBaMM simulation for {n_cycles} cycles (this may take a while)...")
sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
sol = sim.solve()
end_time_pbm = time.time()

t = sol["Time [s]"].entries
eps_s_n = sol["X-averaged negative electrode active material volume fraction"](t)
eps_s_p = sol["X-averaged positive electrode active material volume fraction"](t)
SOH_LAM_n = eps_s_n / eps_s_n[1];
SOH_LAM_p = eps_s_p / eps_s_p[1];
stress_n_t = sol["X-averaged negative particle surface tangential stress [Pa]"](t)
stress_p_t = sol["X-averaged positive particle surface tangential stress [Pa]"](t)
stress_n_r = sol["X-averaged negative particle surface radial stress [Pa]"](t)
stress_p_r = sol["X-averaged positive particle surface radial stress [Pa]"](t)
# --------------------------------------------------------------------------------
# 5. Compute analytical SEI thickness using same parameters
# --------------------------------------------------------------------------------

dt = np.insert(np.diff(t),0,0)

stress_h_min = 0;

stress_h_n = (stress_n_r + 2 * stress_n_t) / 3
stress_h_p = (stress_p_r + 2 * stress_p_t) / 3
j_stress_LAM_n = -Beta_n * ((stress_h_n - stress_h_min) / stress_critical_n) ** m_LAM_n * (stress_h_n > 0) * dt
j_stress_LAM_p = -Beta_p * ((stress_h_p - stress_h_min) / stress_critical_p) ** m_LAM_p * (stress_h_p > 0) * dt

eps_s_n_anal = eps_s_n[1] + np.cumsum(j_stress_LAM_n)
eps_s_p_anal = eps_s_p[1] + np.cumsum(j_stress_LAM_p)
SOH_LAM_n_anal = eps_s_n_anal / eps_s_n_anal[1];
SOH_LAM_p_anal = eps_s_p_anal / eps_s_p_anal[1];

# --------------------------------------------------------------------------------
# 6. Plot results
# --------------------------------------------------------------------------------
plt.figure(figsize=(9, 8))

plt.plot(t / 3600, SOH_LAM_n, label = "NE LAM")
plt.plot(t / 3600, SOH_LAM_p, label =  "PE LAM")
plt.plot(t / 3600, SOH_LAM_n_anal,'--', label = "NE LAM analytic")
plt.plot(t / 3600, SOH_LAM_p_anal,'--', label =  "PE LAM analytic")
plt.ylabel("Loss of active material")
plt.xlabel("Time [h]")
plt.grid(True)

