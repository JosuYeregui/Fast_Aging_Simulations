import pybamm
import matplotlib.pyplot as plt
from pybamm.models.submodels.active_material.base_active_material import BaseModel
from pybamm.models.submodels.active_material.loss_active_material import LossActiveMaterial
import numpy as np

class StressBasedLAM(BaseModel):

    def __init__(self, param, domain):
        super().__init__(param, domain, options=None)

    # def get_fundamental_variables(self):
    #     domain, Domain = self.domain_Domain
    #     eps = pybamm.Variable(f"{Domain} active material fraction")
    #    return {f"{Domain} active material fraction": eps}
    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        eps = pybamm.Variable(f"X-averaged {domain} electrode active material volume fraction",domain="current collector")
        eps = pybamm.PrimaryBroadcast(eps, f"{domain} electrode")
        variables = self._get_standard_active_material_variables(eps)
        return variables# {f"X-averaged {domain} electrode active material volume fraction": eps}

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain

        c_s = variables[f"X-averaged {domain} particle surface concentration [mol.m-3]"]
        c_avg = pybamm.x_average(variables[f"R-averaged {domain} particle concentration [mol.m-3]"])

        Omega = pybamm.Parameter(f"{Domain} electrode partial molar volume [m3.mol-1]")
        E = pybamm.Parameter(f"{Domain} electrode Young's modulus [Pa]")
        nu = pybamm.Parameter(f"{Domain} Poisson's ratio")
        stress_geometric_factor = pybamm.Parameter(f"{Domain} stress geometric factor")

        stress_t = Omega * E * (c_s - c_avg) / (1 - nu) / stress_geometric_factor

        stress_h = (2 * stress_t) / 3

        stress_critial = pybamm.Parameter(f"{Domain} critical stress [Pa]")

        m = pybamm.Parameter(f"{Domain} LAM constant exponential term")
        Beta = pybamm.Parameter(f"{Domain} LAM constant proportional term")

        deps_dt = -Beta * pybamm.AbsoluteValue(stress_h / stress_critial) ** m * (
                    stress_h > 0) # deps_dt = -k * stress_guarro

        variables.update(
            self._get_standard_active_material_change_variables(deps_dt)
        )

        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        eps = variables[f"X-averaged {domain} electrode active material volume fraction"]
        deps_dt = variables[
            f"X-averaged {domain} electrode {phase_name}active material "
            "volume fraction change [s-1]"
        ]

        self.rhs = {eps: deps_dt}

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        # eps = variables[f"{Domain} active material fraction"]
        eps = variables[f"X-averaged {domain} electrode active material volume fraction"]
        eps0 = self.phase_param.epsilon_s # pybamm.Parameter(f"{Domain} initial active material fraction")
        self.initial_conditions = {eps: pybamm.x_average(eps0)}

class StressBasedLAM_FP(LossActiveMaterial):

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain

        c_s = variables[f"X-averaged {domain} particle surface concentration [mol.m-3]"]
        c_avg = pybamm.x_average(variables[f"R-averaged {domain} particle concentration [mol.m-3]"])

        Omega = pybamm.Parameter(f"{Domain} electrode partial molar volume [m3.mol-1]")
        E = pybamm.Parameter(f"{Domain} electrode Young's modulus [Pa]")
        nu = pybamm.Parameter(f"{Domain} Poisson's ratio")
        stress_geometric_factor = pybamm.Parameter(f"{Domain} stress geometric factor")

        stress_t = Omega * E * (c_s - c_avg) / (1 - nu) / stress_geometric_factor

        stress_h = (2 * stress_t) / 3

        stress_critial = pybamm.Parameter(f"{Domain} critical stress [Pa]")

        m = pybamm.Parameter(f"{Domain} LAM constant exponential term")
        Beta = pybamm.Parameter(f"{Domain} LAM constant proportional term")

        deps_dt = -Beta * pybamm.AbsoluteValue(stress_h / stress_critial) ** m * (
                stress_h > 0)  # deps_dt = -k * stress_guarro

        variables.update(
            self._get_standard_active_material_change_variables(deps_dt)
        )

        return variables

# model = pybamm.lithium_ion.SPMe(options={"x-average side reactions":"false", "loss of active material": "current-driven"})

model = pybamm.lithium_ion.SPMe(build=False)
#     options={
#         "loss of active material": "current-driven",
#     }
# )
param_chen = pybamm.ParameterValues("Chen2020")
param = pybamm.ParameterValues("OKane2022")
param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
# param.update(
#     {
#         "Positive electrode current-driven LAM rate": current_LAM,
#         "Negative electrode current-driven LAM rate": current_LAM,
#     },
#     check_already_exists=False,
# )

param.update({
    "Negative initial active material fraction": param["Negative electrode active material volume fraction"],
    "Positive initial active material fraction": param["Positive electrode active material volume fraction"],
    "Negative electrode partial molar volume [m3.mol-1]": param["Negative electrode partial molar volume [m3.mol-1]"],
    "Positive electrode partial molar volume [m3.mol-1]": param["Positive electrode partial molar volume [m3.mol-1]"],
    "Negative electrode Young's modulus [Pa]": param["Negative electrode Young's modulus [Pa]"],
    "Positive electrode Young's modulus [Pa]": param["Positive electrode Young's modulus [Pa]"],
    "Negative critical stress [Pa]": param["Negative electrode critical stress [Pa]"],
    "Positive critical stress [Pa]": param["Positive electrode critical stress [Pa]"],
    "Negative stress geometric factor": 3,
    "Positive stress geometric factor": 3,
    "Negative LAM constant proportional term": param["Negative electrode LAM constant proportional term [s-1]"]/1000.,
    "Positive LAM constant proportional term": param["Positive electrode LAM constant proportional term [s-1]"]/1000.,
    "Negative LAM constant exponential term": param["Negative electrode LAM constant exponential term"],
    "Positive LAM constant exponential term": param["Positive electrode LAM constant exponential term"],
    "Negative Poisson's ratio": param["Negative electrode Poisson's ratio"],
    "Positive Poisson's ratio": param["Positive electrode Poisson's ratio"],
}, check_already_exists=False)

#param["Negative electrode diffusivity [m2.s-1]"] = param["Negative electrode diffusivity [m2.s-1]"](0.5, 298.15)
#param["Positive electrode diffusivity [m2.s-1]"] = param["Positive electrode diffusivity [m2.s-1]"](0.5, 298.15)

model.submodels["Negative active material"] = StressBasedLAM(
    model.param, "Negative"
)

#model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")

model.submodels["Positive active material"] = StressBasedLAM(
    model.param, "Positive"
)
#model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={"loss of active material": "stress-driven"}, x_average=True, phase="primary")
model.build_model()

experiment = pybamm.Experiment([("Charge at 1.5 A until 4.2 V", "Rest for 1 hour", "Discharge at 1 A until 2.5 V")]*100)

sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)

sol = sim.solve(initial_soc=0.)

# Get surface and average negative electrode concentrations
#c_ave = sol.cycles[0]["Average negative particle concentration [mol.m-3]"].entries
c_ave = np.mean(sol.cycles[0]["R-averaged negative particle concentration [mol.m-3]"].entries, axis=0)
c_surf = sol.cycles[0]["X-averaged negative particle surface concentration [mol.m-3]"].entries
c_ave_p = sol.cycles[0]["Average positive particle concentration [mol.m-3]"].entries
c_surf_p = sol.cycles[0]["X-averaged positive particle surface concentration [mol.m-3]"].entries
# lam = sol["X-averaged negative electrode active material volume fraction"].entries
# lam = sol["Negative active material fraction"].entries
# lam = sol["Negative electrode active material volume fraction"].entries
lam = sol.cycles[0]["X-averaged positive electrode active material volume fraction"].entries
lam_2 = sol.cycles[0]["X-averaged negative electrode active material volume fraction"].entries
LAM_true = sol.cycles[0]["Negative electrode capacity [A.h]"].entries
I = sol.cycles[0]["Current [A]"].entries

fig, ax = plt.subplots()
ax.plot(c_surf)
ax.plot(c_ave)
ax2 = ax.twinx()
ax2.plot(c_ave - c_surf, color="red")
plt.show()

fig, ax = plt.subplots()
ax.plot(c_surf_p)
ax.plot(c_ave_p)
ax2 = ax.twinx()
ax2.plot(c_ave_p - c_surf_p, color="red")
plt.show()

fig, ax = plt.subplots()
ax.plot(lam, label="LAM_p")
#ax.plot(lam_2, label="LAM_n")
ax.set_xlabel("Index")
ax.set_ylabel("lam", color="blue")
ax.legend()
ax2 = ax.twinx()
ax2.plot(LAM_true, color="red")
ax2.set_ylabel("Q", color="red")
plt.show()

fig, ax = plt.subplots()
#ax.plot(lam, label="LAM_p")
ax.plot(lam_2, label="LAM_n")
ax.set_xlabel("Index")
ax.set_ylabel("lam", color="blue")
ax.legend()
ax2 = ax.twinx()
ax2.plot(LAM_true, color="red")
ax2.set_ylabel("Q", color="red")
plt.show()

c_ave = sol.cycles[-1]["Average negative particle concentration [mol.m-3]"].entries
c_surf = sol.cycles[-1]["X-averaged negative particle surface concentration [mol.m-3]"].entries
c_ave_p = sol.cycles[-1]["Average positive particle concentration [mol.m-3]"].entries
c_surf_p = sol.cycles[-1]["X-averaged positive particle surface concentration [mol.m-3]"].entries
# lam = sol["X-averaged negative electrode active material volume fraction"].entries
# lam = sol["Negative active material fraction"].entries
# lam = sol["Negative electrode active material volume fraction"].entries
lam = sol.cycles[-1]["X-averaged positive electrode active material volume fraction"].entries
lam_2 = sol.cycles[-1]["X-averaged negative electrode active material volume fraction"].entries
LAM_true = sol.cycles[-1]["Negative electrode capacity [A.h]"].entries
I = sol.cycles[-1]["Current [A]"].entries

fig, ax = plt.subplots()
ax.plot(c_surf)
ax.plot(c_ave)
ax2 = ax.twinx()
ax2.plot(c_ave - c_surf, color="red")
plt.show()

fig, ax = plt.subplots()
ax.plot(c_surf_p)
ax.plot(c_ave_p)
ax2 = ax.twinx()
ax2.plot(c_ave_p - c_surf_p, color="red")
plt.show()

fig, ax = plt.subplots()
ax.plot(lam, label="LAM_p")
#ax.plot(lam_2, label="LAM_n")
ax.set_xlabel("Index")
ax.set_ylabel("lam", color="blue")
ax.legend()
ax2 = ax.twinx()
ax2.plot(LAM_true, color="red")
ax2.set_ylabel("Q", color="red")
plt.show()

fig, ax = plt.subplots()
#ax.plot(lam, label="LAM_p")
ax.plot(lam_2, label="LAM_n")
ax.set_xlabel("Index")
ax.set_ylabel("lam", color="blue")
ax.legend()
ax2 = ax.twinx()
ax2.plot(LAM_true, color="red")
ax2.set_ylabel("Q", color="red")
plt.show()