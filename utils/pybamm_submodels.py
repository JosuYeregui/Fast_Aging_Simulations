import pybamm
from pybamm.models.submodels.active_material.base_active_material import BaseModel
from pybamm.models.submodels.active_material.loss_active_material import LossActiveMaterial
from pybamm.models.submodels.interface.lithium_plating.plating import Plating

def update_LAM_params(param):

    param.update({
        "Negative initial active material fraction": param["Negative electrode active material volume fraction"],
        "Positive initial active material fraction": param["Positive electrode active material volume fraction"],
        "Negative electrode partial molar volume [m3.mol-1]": param[
            "Negative electrode partial molar volume [m3.mol-1]"],
        "Positive electrode partial molar volume [m3.mol-1]": param[
            "Positive electrode partial molar volume [m3.mol-1]"],
        "Negative electrode Young's modulus [Pa]": param["Negative electrode Young's modulus [Pa]"],
        "Positive electrode Young's modulus [Pa]": param["Positive electrode Young's modulus [Pa]"],
        "Negative critical stress [Pa]": param["Negative electrode critical stress [Pa]"],
        "Positive critical stress [Pa]": param["Positive electrode critical stress [Pa]"],
        "Negative stress geometric factor": 3,
        "Positive stress geometric factor": 3,
        "Negative LAM constant proportional term": param["Negative electrode LAM constant proportional term [s-1]"],
        "Positive LAM constant proportional term": param["Positive electrode LAM constant proportional term [s-1]"],
        "Negative LAM constant exponential term": param["Negative electrode LAM constant exponential term"],
        "Positive LAM constant exponential term": param["Positive electrode LAM constant exponential term"],
        "Negative Poisson's ratio": param["Negative electrode Poisson's ratio"],
        "Positive Poisson's ratio": param["Positive electrode Poisson's ratio"],
    }, check_already_exists=False)

    return param

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
            self._get_standard_active_material_change_variables(deps_dt),
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
        phase_name = self.phase_name

        c_s = variables[f"X-averaged {domain} {phase_name}particle surface concentration [mol.m-3]"]
        #c_avg = pybamm.x_average(variables[f"R-averaged {domain} {phase_name}particle concentration [mol.m-3]"])
        c_avg = variables[f"Average {domain} {phase_name}particle concentration [mol.m-3]"]

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
                stress_h > 0)

        variables.update(
            self._get_standard_active_material_change_variables(deps_dt)
        )

        return variables

# class Plating_FP_ForTesting(Plating):
#
#     def get_coupled_variables(self, variables):
#         phase_param = self.phase_param
#         domain, Domain = self.domain_Domain
#         T = variables[f"{Domain} electrode temperature [K]"]
#         alpha_plating = phase_param.alpha_plating
#         c_plated_Li = variables[
#             f"{Domain} {self.phase_name}lithium plating concentration [mol.m-3]"
#         ]
#         F_RT = self.param.F / (self.param.R * T)
#
#         ce = variables[
#             f"{Domain} {self.phase_name}electrolyte concentration [mol.m-3]"
#         ]
#         j0_plating = phase_param.j0_plating(ce, c_plated_Li, T)
#         i_app = variables["Total current density [A.m-2]"]
#
#         c_s = variables[f"X-averaged {domain} {self.phase_name}particle surface concentration [mol.m-3]"]
#         c_s_max = self.phase_param.c_max
#         k_int = 6.48e-7
#         E_r = 35000
#         arrhenius = pybamm.exp(E_r / self.param.R * (1 / 298.15 - 1 / T))
#         j0_int = k_int * pybamm.sqrt(ce) * pybamm.sqrt(c_s) * pybamm.sqrt(c_s_max - c_s) * arrhenius
#
#         eta_r = -2 / F_RT * pybamm.arcsinh(i_app / (self.phase_param.a_typ * j0_int * self.param.n.L))
#
#         ocp = variables[
#             f"{Domain} electrode open-circuit potential [V]"
#         ]
#         eta_plating = ocp - eta_r
#
#         j_stripping = -j0_plating * pybamm.exp(-F_RT * alpha_plating * eta_plating)
#
#         variables.update(self._get_standard_overpotential_variables(eta_plating))
#         variables.update(self._get_standard_reaction_variables(j_stripping))
#
#         # Add other standard coupled variables
#         # variables.update(super().get_coupled_variables(variables))
#
#         j_plating_av = variables[
#             f"X-averaged {domain} electrode {self.phase_name}lithium plating "
#             "interfacial current density [A.m-2]"
#         ]
#         j_plating = variables[
#             f"{Domain} electrode {self.phase_name}lithium plating "
#             "interfacial current density [A.m-2]"
#         ]
#         variables.update(
#             {
#                 f"X-averaged {domain} electrode {self.phase_name}lithium plating "
#                 "interfacial current density [A.m-2]": j_plating_av,
#                 f"{Domain} electrode {self.phase_name}lithium plating "
#                 "interfacial current density [A.m-2]": j_plating,
#             }
#         )
#
#         variables.update(
#             self._get_standard_volumetric_current_density_variables(variables)
#         )
#
#         return variables
#
#
#     def set_rhs(self, variables):
#         domain, Domain = self.domain_Domain
#         phase_name = self.phase_name
#
#         c_plated_Li = variables[
#             f"X-averaged {domain} {phase_name}lithium plating concentration "
#             "[mol.m-3]"
#         ]
#         c_dead_Li = variables[
#             f"X-averaged {domain} {phase_name}dead lithium concentration [mol.m-3]"
#         ]
#         j_stripping = variables[
#             f"X-averaged {domain} electrode {self.phase_name}lithium plating "
#             "interfacial current density [A.m-2]"
#         ]
#
#         a = pybamm.x_average(variables[
#             f"{Domain} electrode {phase_name}surface area to volume ratio [m-1]"
#         ])
#
#         dc_plated_Li = -a * j_stripping / self.param.F
#         dc_dead_Li = dc_plated_Li
#
#         self.rhs = {
#             c_plated_Li: dc_plated_Li,
#             c_dead_Li: dc_dead_Li,
#         }
#
#     def set_initial_conditions(self, variables):
#         domain, Domain = self.domain_Domain
#         phase_name = self.phase_name
#         if self.x_average is True:
#             c_plated_Li = variables[
#                 f"X-averaged {domain} {phase_name}lithium plating concentration "
#                 "[mol.m-3]"
#             ]
#             c_dead_Li = variables[
#                 f"X-averaged {domain} {phase_name}dead lithium concentration [mol.m-3]"
#             ]
#         else:
#             c_plated_Li = variables[
#                 f"{Domain} {phase_name}lithium plating concentration [mol.m-3]"
#             ]
#             c_dead_Li = variables[
#                 f"{Domain} {phase_name}dead lithium concentration [mol.m-3]"
#             ]
#         c_dead_Li_0 = self.phase_param.c_plated_Li_0
#         c_plated_Li_0 = pybamm.Scalar(0.) # self.phase_param.c_plated_Li_0
#
#         self.initial_conditions = {c_plated_Li: c_plated_Li_0, c_dead_Li: c_dead_Li_0}


class Plating_FP_old(Plating):

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        if self.x_average is True:
            c_plated_Li = variables[
                f"X-averaged {domain} {phase_name}lithium plating concentration "
                "[mol.m-3]"
            ]
            c_dead_Li = variables[
                f"X-averaged {domain} {phase_name}dead lithium concentration [mol.m-3]"
            ]
        else:
            c_plated_Li = variables[
                f"{Domain} {phase_name}lithium plating concentration [mol.m-3]"
            ]
            c_dead_Li = variables[
                f"{Domain} {phase_name}dead lithium concentration [mol.m-3]"
            ]
        c_dead_Li_0 = self.phase_param.c_plated_Li_0
        c_plated_Li_0 = pybamm.Scalar(0.) # self.phase_param.c_plated_Li_0

        self.initial_conditions = {c_plated_Li: c_plated_Li_0, c_dead_Li: c_dead_Li_0}