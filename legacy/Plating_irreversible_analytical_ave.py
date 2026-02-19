# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 15:53:27 2025

@author: ilopetegui
"""

import numpy as np
import matplotlib.pyplot as plt
import pybamm

# --------------------------------------------------------------------------------
# 1. Model setup
# --------------------------------------------------------------------------------
model = pybamm.lithium_ion.SPMe(
    options={"lithium plating": "irreversible"}  # enable Li plating model
)

param = pybamm.ParameterValues("OKane2022")
#param.update({"Ambient temperature [K]": 298})
param.update({"Upper voltage cut-off [V]": 4.21})

experiment = pybamm.Experiment([
    "Discharge at 1C until 2.5 V",
    "Charge at 2C until 4.2 V",
    "Hold at 4.2 V until C/20"
] * 10, period="1 second")

sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
print("Running simulation...")
sim.solve()
sol = sim.solution
t = sol["Time [s]"].entries

# Extract plating variables
eta_key = "Negative electrode lithium plating reaction overpotential [V]"
j_key = "Negative electrode lithium plating interfacial current density [A.m-2]"
delta_phi = "Negative electrode surface potential difference [V]"
eta_sei = "Negative electrode SEI film overpotential [V]"
eta_t = sol[delta_phi](t)
#eta_t += sol[eta_sei](t)

x_n = sol["x_n [m]"](0)

eta_sei = sol[
            f"X-averaged negative electrode SEI film overpotential [V]"
        ](t)

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
c_e = sol["Negative electrolyte concentration [mol.m-3]"](t)
k_plating = param["Lithium plating kinetic rate constant [m.s-1]"]
i0 = k_plating*F*c_e
#i0 = 1e-4*c_e

#i0 = 5.2e-7*a*c_e/1000 # Yang2017

#i0_py = param["Exchange-current density for plating [A.m-2]"]
#i0_val = i0_py(c_e,c_Li,T)

# --------------------------------------------------------------------------------
# 3. Analytical piecewise integration
# --------------------------------------------------------------------------------
def integrate_piecewise(t, eta_t, i0, alpha_c, F, R, T, V_Li, L0=0.0, slope_tol=1e-15):
    """Piecewise integrate the plating rate for η(t)."""
    k_t = - i0 * np.exp(-alpha_c * F * eta_t / (R * T)) # A/m2 effective rate

    k_t = np.trapezoid(np.transpose(k_t), x_n) / x_n[-1]

    ax, fig = plt.subplots()
    fig.plot(t/3600, k_t, label="k_t")
    fig.plot(t/3600, j_pl_num, label="j_pl_num")
    fig.legend()
    plt.show()
    #k_t = j_pl_num
    dk = np.gradient(k_t, t)
    flat = np.abs(dk / np.maximum(np.abs(k_t), 1e-30)) < slope_tol

    L = np.zeros_like(t)
    L[0] = L0
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        if flat[i]:
            k_mean = np.mean(k_t[max(0, i-3):i+4])  # local avg
        else:
            k_mean = 0.5 * (k_t[i] + k_t[i+1])
        L[i+1] = L[i] - V_Li/F * k_mean * dt
    return L, k_t

[L_anal,j_anal] = integrate_piecewise(t, eta_t, i0, alpha_c, F, R, T, V_Li)

# Numerical PyBaMM result for comparison
L_num = V_Li / a * sol["X-averaged negative dead lithium concentration [mol.m-3]"](t)
L_num_dead = sol["X-averaged negative dead lithium thickness [m]"](t) #Gorakuan berdina
LLI = sol["Loss of lithium inventory [%]"](t)
# --------------------------------------------------------------------------------
# 4. Plot comparison
# --------------------------------------------------------------------------------
# plt.figure(figsize=(8,6))
# plt.plot(t/3600, j_pl_num, label="PyBaMM numerical")
# plt.plot(t/3600, j_anal, '--', label="Analytical (piecewise)")
# plt.xlabel("Time [h]")
# plt.ylabel("Lithium plating interfacial current density [A.m-2]")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,6))
# plt.plot(t/3600, L_num_dead, label="PyBaMM numerical")
# plt.plot(t/3600, L_anal, '--', label="Analytical (piecewise)")
# plt.xlabel("Time [h]")
# plt.ylabel("Lithium plating thickness [m]")
# plt.legend()
# plt.grid(True)
# plt.title("Lithium plating growth: analytical vs numerical")
# plt.show()


plt.figure(figsize=(9, 8))

ax1 = plt.subplot(311)
ax1.plot(t / 3600, eta_t[0,:])
# ax1.plot(t / 3600, delta_phi_num)
ax1.set_ylabel("Neg. electrode plating overpotential (V)")
ax1.grid(True)

ax2 = plt.subplot(312, sharex=ax1)
ax2.plot(t/3600, j_pl_num, label="PyBaMM numerical")
ax2.plot(t/3600, j_anal,'--', label="Analytical (piecewise)")

ax3 = plt.subplot(313, sharex=ax1)
ax3.plot(t/3600, L_num_dead, label="PyBaMM numerical")
ax3.plot(t/3600, L_anal,'--', label="Analytical (piecewise)")

plt.show()


