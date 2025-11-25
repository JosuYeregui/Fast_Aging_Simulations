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
param = pybamm.ParameterValues("OKane2022")

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
E_p = param["Positive electrode Young\'s modulus [Pa]"]
nu_p = param["Positive electrode Poisson\'s ratio"]
Omega_p = param["Positive electrode partial molar volume [m3.mol-1]"]
E_n = param["Negative electrode Young\'s modulus [Pa]"]
nu_n = param["Negative electrode Poisson\'s ratio"]
Omega_n = param["Negative electrode partial molar volume [m3.mol-1]"]

# --------------------------------------------------------------------------------
# 3. Build experiment: 50 charge/discharge cycles
# --------------------------------------------------------------------------------
n_cycles = 1
steps = []
for _ in range(n_cycles):
    steps.append("Discharge at 2 C until 2.5 V")
    steps.append("Charge at 1 C until 4.2 V")
    steps.append("Hold at 4.2 V until 0.02 C")
    # steps.append("Rest for 10 h")
    
# for _ in range(n_cycles):
#     steps.append("Charge at 0.5 C until 4.2 V")
#     steps.append("Hold at 4.2 V until 0.02 C")
#     steps.append("Discharge at 0.5 C until 2.5 V")
    
    
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

cs_avg_n = sol["Average negative particle concentration [mol.m-3]"](t)
cs_avg_p = sol["Average positive particle concentration [mol.m-3]"](t)
c_surf_n = sol["X-averaged negative particle surface concentration [mol.m-3]"](t)
c_surf_p = sol["X-averaged positive particle surface concentration [mol.m-3]"](t)

L_n = param["Negative electrode thickness [m]"]
L_p = param["Positive electrode thickness [m]"]
Aelec = param["Electrode width [m]"] * param["Electrode height [m]"]
epss_n = param["Negative electrode active material volume fraction"]
epss_p = param["Positive electrode active material volume fraction"]
Rs_n = param["Negative particle radius [m]"]
Rs_p = param["Positive particle radius [m]"]
as_n = epss_n*3/Rs_n
as_p = epss_p*3/Rs_p
Ds_n = 3.9e-14*100 #param["Negative electrode diffusivity [m2.s-1]"]
Ds_p = 5.387e-15*100 #param["Positive electrode diffusivity [m2.s-1]"]
Ds_n = 3.3e-14
Ds_p = 4e-13
# Ds_p_fun = param["Positive electrode diffusivity [m2.s-1]"];
# Ds_p = Ds_p_fun(0.5,298.15)

I_single = np.concatenate([
    np.ones(1800) * 8,
    -2 * np.ones(5300)
])

I = np.tile(I_single, n_cycles)  # repeat waveform n_cycles times

# I = sol["Current [A]"](t)
# j_p_sol = sol["X-averaged positive electrode interfacial current density [A.m-2]"](t)
# j_n_sol = sol["X-averaged negative electrode interfacial current density [A.m-2]"](t)

j_n = I / as_n / F / L_n / Aelec
j_p = - I / as_p / F / L_p / Aelec

cs_diff_n = j_n * Rs_n / 5 / Ds_n #cs_avg_n - c_surf_n
cs_diff_p = j_p * Rs_p / 5 / Ds_p #cs_avg_n - c_surf_n
stress_geometric_factor = 3
stress_n_t = Omega_n*E_n*(cs_diff_n) / (1 - nu_n) / stress_geometric_factor
stress_p_t = Omega_p*E_p*(cs_diff_p) / (1 - nu_p) / stress_geometric_factor

stress_n_r = 0
stress_p_r = 0


stress_h_n = (stress_n_r + 2 * stress_n_t) / 3
stress_h_p = (stress_p_r + 2 * stress_p_t) / 3
j_stress_LAM_n = -Beta_n * ((stress_h_n[0:len(t)] - stress_h_min) / stress_critical_n) ** m_LAM_n * (stress_h_n[0:len(t)] > 0) * dt
j_stress_LAM_p = -Beta_p * ((stress_h_p[0:len(t)] - stress_h_min) / stress_critical_p) ** m_LAM_p * (stress_h_p[0:len(t)] > 0) * dt

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


plt.figure(figsize=(9, 8))
ax1 = plt.subplot(311)
ax1.plot(t / 3600, cs_avg_n, label = "Avg")
ax1.plot(t / 3600, c_surf_n, label =  "Surf")
ax1.set_ylabel("NE concentrations")
ax1.set_xlabel("Time [h]")
ax1.grid(True)

ax2 = plt.subplot(312)
ax2.plot(t / 3600, cs_avg_p, label = "Avg")
ax2.plot(t / 3600, c_surf_p, label =  "Surf")
ax2.set_ylabel("PE concentrations")
ax2.set_xlabel("Time [h]")
ax2.grid(True)

ax3 = plt.subplot(313)
ax3.plot(t / 3600, cs_avg_n - c_surf_n, label = "Neg")
ax3.plot(t / 3600, cs_avg_p - c_surf_p, label =  "Pos")
ax3.plot(t/3600, cs_diff_n[0:len(t)])
ax3.plot(t/3600, cs_diff_p[0:len(t)])
ax3.set_ylabel("PE concentrations")
ax3.set_xlabel("Time [h]")
ax3.grid(True)
plt.show()

