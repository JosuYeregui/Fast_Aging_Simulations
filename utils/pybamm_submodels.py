import pybamm
from pybamm.models.submodels.active_material.base_active_material import BaseModel
from pybamm.models.submodels.active_material.loss_active_material import LossActiveMaterial

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