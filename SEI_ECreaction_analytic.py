# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 20:03:44 2025

@author: Iker Lopetegi
"""

import numpy as np
import matplotlib.pyplot as plt
import pybamm
import time

# --------------------------------------------------------------------------------
# 1. Build PyBaMM SPMe model with interstitial-diffusion-limited SEI
# --------------------------------------------------------------------------------
print("Building SPMe with interstitial-diffusion limited SEI option...")
model = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited","x-average side reactions":"false"})
param = pybamm.ParameterValues("Chen2020")

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


# --------------------------------------------------------------------------------
# 3. Build experiment: 50 charge/discharge cycles
# --------------------------------------------------------------------------------
n_cycles = 10
steps = []
for _ in range(n_cycles):
    steps.append("Charge at 1 C until 4.2 V")
    steps.append("Hold at 4.2 V until 0.02 C")
    steps.append("Discharge at 0.5 C until 2.5 V")
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
L_num = sol["X-averaged negative SEI thickness [m]"](t)

# --------------------------------------------------------------------------------
# 5. Compute analytical SEI thickness using same parameters
# --------------------------------------------------------------------------------

# evaluate eta(t) at solver times (use the same t as sol)
phi_s = sol["X-averaged negative electrode potential [V]"](t)   # may be spatial -> average same way PyBaMM does
phi_e = sol["X-averaged negative electrolyte potential [V]"](t)
delta_phi = sol["X-averaged negative electrode surface potential difference [V]"](t)
eta = phi_s - phi_e - U_sei

eta = sol["X-averaged negative electrode open-circuit potential [V]"](t) - U_sei


"""
We need a method to obtain eta without computing the PBM. We should estimate the
SOC points. In this case, if we consider full charge-discharges for 1 h charge
and 2h discharge...
"""
t_ch = range(3600)
t_dch = range(7200)
Uocp_p = param["Positive electrode OCP [V]"]
Uocp_n = param["Negative electrode OCP [V]"]
SOL_n_cycle = np.append(np.linspace(0.03, 0.9, 3601),np.linspace(0.9, 0.03, 7201))


SOL_n = np.tile(SOL_n_cycle, n_cycles*2)
eta = Uocp_n(SOL_n) - U_sei

plt.figure(figsize=(9, 8))
plt.plot(eta)




# eta = delta_phi - U_sei - sol["X-averaged SEI film overpotential [V]"](t)
exp_vals = np.exp(-alpha * F * eta / (R * T))
# I = np.zeros_like(t)
# I[1:] = np.cumsum(0.5*(expvals[1:]+expvals[:-1])*(t[1:]-t[:-1]))
# I = np.zeros_like(np.tile(t,(len(exp_vals),1)))
# I[:,1:] = np.cumsum(0.5/2 * (exp_vals[:,1:] + exp_vals[:,:-1]) * (t[1:] - t[:-1]),axis = 1)
# L_anal = np.sqrt(L0**2 + 2 * V_sei * k_sei * D_ec * c_ec * I)
k_exp = exp_vals*k_sei
A = c_ec * V_sei / 2
# L_anal = (-D_ec + np.sqrt((D_ec + k_exp * L0)**2 + 2 * A * D_ec * k_exp ** 2 * t)) / k_exp

def piecewise_integration(t, k_exp_t, D, A, L0, slope_tol=1e-3, min_len=5):
    """
    Piecewise integration for dL/dt = A*D*k/(D + k*L)
    treating nearly-constant k segments analytically.
    """
    t = np.asarray(t)
    k_exp_t = np.asarray(k_exp_t)
    dt = np.diff(t)

    # Compute relative slope of k_exp_t to detect flat regions ("plateaus")
    dk = np.gradient(k_exp_t, t,)
    flat = np.abs(dk / np.maximum(np.abs(k_exp_t), 1e-30)) < slope_tol

    # Identify contiguous flat regions
    segments = []
    i = 0
    N = len(t)
    while i < N - 1:
        if flat[i]:
            j = i
            while j < N - 1 and flat[j]:
                j += 1
            if j - i >= min_len:
                segments.append((i, j))
            i = j
        else:
            i += 1

    # Integrate step by step
    L = np.zeros_like(t)
    L[0] = L0
    seg_idx = 0

    i = 0
    while i < N - 1:
        # If this point is inside a plateau segment
        if seg_idx < len(segments) and segments[seg_idx][0] <= i < segments[seg_idx][1]:
            i_start, i_end = segments[seg_idx]
            k_plateau = np.mean(k_exp_t[i_start:i_end+1])
            dt_plateau = t[i_end] - t[i_start]
            Li = L[i_start]
            # Exact constant-k update
            L_next = (-D + np.sqrt(D**2 + 2*k_plateau*(D*Li + 0.5*k_plateau*Li**2 + A*D*k_plateau*dt_plateau))) / k_plateau
            L[i_start+1:i_end+1] = L_next
            i = i_end
            seg_idx += 1
        else:
            # Normal small-step exact update with local k
            k_mid = 0.5 * (k_exp_t[i] + k_exp_t[i+1])
            L[i+1] = (-D_ec + np.sqrt(D_ec**2 + 2*k_mid*(D_ec*L[i] + 0.5*k_mid*L[i]**2 + A*D_ec*k_mid*dt[i]))) / k_mid
            # L_anal = (-D_ec + np.sqrt((D_ec + k_exp * L0)**2 + 2 * A * D_ec * k_exp ** 2 * t)) / k_exp

            i += 1
    return L, flat, segments

start_time_piecewise = time.time()
L_anal, flat, segments = piecewise_integration(t, k_exp[0:len(t)], D_ec, A, L0)
# L_anal_mean = np.mean(L_anal, axis=0)
end_time_piecewise = time.time()


# --------------------------------------------------------------------------------
# 6. Plot results
# --------------------------------------------------------------------------------
plt.figure(figsize=(9, 8))

ax1 = plt.subplot(311)
ax1.plot(t / 3600, eta[0:len(t)])
# ax1.plot(t / 3600, delta_phi_num)
ax1.set_ylabel("Neg. electrode SEI overpotential (V)")
ax1.grid(True)

ax2 = plt.subplot(312, sharex=ax1)
# ax2.plot(t / 3600, L_num, label="PyBaMM numerical (interstitial-diffusion SEI)")
# for i in range(L_anal.shape[0]):
#    plt.plot(t/3600, L_anal[i,:])
# for i in range(L_num.shape[0]):
#    plt.plot(t/3600, L_num[i,:],'--')
      
plt.plot(t/3600, L_num, label="Numerical (ec-reaction SEI)")
ax2.plot(t/3600, L_anal,"--", label="Analytical")
ax2.set_ylabel("SEI thickness L (m)")
ax2.legend()
ax2.grid(True)
ax2.set_title("SEI growth comparison")

ax3 = plt.subplot(313, sharex=ax1)
rel_err = np.abs(L_num - L_anal) / np.maximum(L_anal, 1e-20)
# for i in range(L_num.shape[0]):
#    plt.plot(t/3600, L_anal[i,:],)
ax3.plot(t[0:len(eta)] / 3600, rel_err)
ax3.set_xlabel("Time (h)")
ax3.set_ylabel("Relative error")
ax3.set_yscale("log")
ax3.grid(True)
ax3.set_title("Relative error |L_num - L_anal| / L_anal")

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------
# 7. Print summary
# --------------------------------------------------------------------------------
print("\nSimulation summary:")
print(f"  Final numerical L = {L_num[-1]:.3e} m")
print(f"  Final analytical L = {L_anal[-1]:.3e} m")
print(f"  Max relative error = {np.nanmax(rel_err):.3e}")
print("---PyBaMM needs %s seconds ---" % (end_time_pbm - start_time_pbm))
print("---Piecewise needs %s seconds ---" % (end_time_piecewise - start_time_piecewise))



