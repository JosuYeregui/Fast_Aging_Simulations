# ============================================================
# params_io.py
# ============================================================

import yaml
import numpy as np
import os
import pybamm

from utils.characterize import get_Q, get_Rint, get_OVC_func

# YAML utilities

def _save_yaml(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        yaml.safe_dump(data, f)


def _load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


# WRITE ALL PARAMETERS TO ONE YAML FILE

def write_all_params_to_yaml(param, model, U_fun_name, yaml_file):
    data = {
        "global": {},
        "model": {},
        "LAM": {},
        "SEI": {},
        "Plating": {}
    }

    # ---------------- Global ----------------
    data["global"]["A"] = param["Electrode height [m]"] * param["Electrode width [m]"]
    data["global"]["F"] = 96485.33212
    data["global"]["V_max"] = param["Upper voltage cut-off [V]"]
    data["global"]["V_min"] = param["Lower voltage cut-off [V]"]
    data["global"]["Q_nom"] = param["Nominal cell capacity [A.h]"]

    # ---------------- LAM ----------------
    names = {"Positive": "p", "Negative": "n"}
    for k, d in names.items():
        data["LAM"][d] = {
            "Omega": param[k + " electrode partial molar volume [m3.mol-1]"],
            "omega_fact": 3,
            "E": param[k + " electrode Young's modulus [Pa]"],
            "nu": param[k + " electrode Poisson's ratio"],
            "stress_critical": param[k + " electrode critical stress [Pa]"],
            "Beta": param[k + " electrode LAM constant proportional term [s-1]"],
            "m": param[k + " electrode LAM constant exponential term"],
            "D": param[k + " electrode diffusivity [m2.s-1]"],
            "R": param[k + " particle radius [m]"],
            "L": param[k + " electrode thickness [m]"],
            "csmax": param["Maximum concentration in " + k.lower() + " electrode [mol.m-3]"],
            "cini": param["Initial concentration in " + k.lower() + " electrode [mol.m-3]"],
            "U": U_fun_name+d,
            "eps_0": param[k + " electrode active material volume fraction"]
        }

    # ---------------- SEI ----------------
    sei = data["SEI"]
    sei["k_sei"] = param["SEI kinetic rate constant [m.s-1]"]
    sei["D_ec"] = param["EC diffusivity [m2.s-1]"]
    sei["c_ec"] = param["EC initial concentration in electrolyte [mol.m-3]"]
    sei["alpha"] = 0.5#param["SEI charge transfer coefficient"]
    sei["U_sei"] = param["SEI open-circuit potential [V]"]
    sei["R"] = 8.314462 # param["Ideal gas constant [J.mol-1.K-1]"]
    sei["T"] = param["Ambient temperature [K]"]
    sei["V_SEI"] = param["SEI partial molar volume [m3.mol-1]"]
    sei["z_SEI"] = param["Ratio of lithium moles to SEI moles"]
    sei["L_0"] = param["Initial SEI thickness [m]"]
    sei["rho_SEI"] = param["SEI resistivity [Ohm.m]"]
    sei["A_ec"] = sei["c_ec"] * sei["V_SEI"] / sei["z_SEI"]
    sei["E_act_SEI"] = param["SEI growth activation energy [J.mol-1]"]

    # ----------------- Plating ----------------
    plating = data["Plating"]
    plating["k_plating"] = param["Lithium plating kinetic rate constant [m.s-1]"]
    plating["alpha_plating"] = param["Lithium plating transfer coefficient"]
    plating["ce_0"] = param["Initial concentration in electrolyte [mol.m-3]"]# param["Initial concentration of lithium in electrolyte [mol.m-3]"]
    plating["t_plus"] = param["Cation transference number"] # param["Lithium transference number"]
    brug_n = param["Negative electrode Bruggeman coefficient (electrolyte)"]
    por_n = param["Negative electrode porosity"]
    brug_s = param["Separator Bruggeman coefficient (electrolyte)"]
    por_s = param["Separator porosity"]
    plating["Deff_n"] = float(param["Electrolyte diffusivity [m2.s-1]"](1., 298.15).value * por_n ** brug_n)
    plating["Deff_s"] = float(param["Electrolyte diffusivity [m2.s-1]"](1., 298.15).value * por_s ** brug_s)
    # plating["Deff_n"] = float(param["Electrolyte diffusivity [m2.s-1]"].value * por_n ** brug_n)
    # plating["Deff_s"] = float(param["Electrolyte diffusivity [m2.s-1]"].value * por_s ** brug_s)
    plating["L_s"] = float(param["Separator thickness [m]"])
    plating["m_ref"] = 6.48e-7  # From Okane2022
    plating["V_Li"] = float(param["Lithium metal partial molar volume [m3.mol-1]"])

    # ---------------- Battery model ----------------
    x_0, x_100, y_100, y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(param)

    data["global"]["x_0"] = float(x_0)
    data["global"]["x_100"] = float(x_100)
    data["global"]["y_100"] = float(y_100)
    data["global"]["y_0"] = float(y_0)

    eps_n = data["LAM"]["n"]["eps_0"]
    L = data["LAM"]["n"]["L"]
    A = data["global"]["A"]
    cs_max = data["LAM"]["n"]["csmax"]
    F = data["global"]["F"]

    data["model"]["Q"] = float(eps_n * L * cs_max * F * A * (x_100 - x_0) / 3600.)
    data["model"]["R_int"] = float(get_Rint(model, param))

    _save_yaml(data, yaml_file)

# OCP functions for Chen2020
FUNCTION_REGISTRY = {}
def register_function(name):
    def decorator(fn):
        FUNCTION_REGISTRY[name] = fn
        return fn
    return decorator

param_chen = pybamm.ParameterValues("Chen2020")
@register_function("Chen2020_U_n")
def Chen2020_U_n(sol):
    return param_chen["Negative electrode OCP [V]"](sol)

@register_function("Chen2020_U_p")
def Chen2020_U_p(sol):
    return param_chen["Positive electrode OCP [V]"](sol)

param_mohtat = pybamm.ParameterValues("Mohtat2020")
@register_function("Mohtat2020_U_n")
def Mohtat2020_U_n(sol):
    return param_mohtat["Negative electrode OCP [V]"](sol)

@register_function("Mohtat2020_U_p")
def Mohtat2020_U_p(sol):
    return param_mohtat["Positive electrode OCP [V]"](sol)

# LOADERS

def load_params(yaml_file, load_LAM=True, load_SEI=True, load_plating=True):
    """
    Load parameters from YAML and attach functions from registry.
    """
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)

    input_vars = {}

    _load_global(cfg, input_vars)
    if load_LAM:
        _load_LAM(cfg, input_vars)
    if load_SEI:
        _load_SEI(cfg, input_vars)
    if load_plating:
        _load_plating(cfg, input_vars)

    return input_vars

# ---------------- Global ----------------
def _load_global(cfg, input_vars):
    input_vars.update(cfg.get("global", {}))
    input_vars.update(cfg.get("model", {}))

# ---------------- LAM ----------------
def _load_LAM(cfg, input_vars):
    for d in ("p", "n"):
        lam = cfg["LAM"][d]

        for k, v in lam.items():
            if isinstance(v, str):
                # map function name to actual Python function
                input_vars[f"{k}_{d}"] = FUNCTION_REGISTRY[v]
            else:
                input_vars[f"{k}_{d}"] = v

        # Reattach cs_cave functions
        def make_cs_cave(domain):
            def cs_cave(I, eps):
                as_particle = eps * 3 / input_vars[f"R_{domain}"]
                I_eff = -I
                if domain == "n":
                    j = I_eff / (
                        input_vars["F"] * as_particle * input_vars[f"L_{domain}"] * input_vars["A"]
                    )
                else:
                    j = -I_eff / (
                        input_vars["F"] * as_particle * input_vars[f"L_{domain}"] * input_vars["A"]
                    )
                return -j / 5 * input_vars[f"R_{domain}"] / input_vars[f"D_{domain}"]
            return cs_cave

        input_vars[f"cs_cave_{d}"] = make_cs_cave(d)

# ---------------- SEI ----------------
def _load_SEI(cfg, input_vars):
    sei = cfg["SEI"]
    for k, v in sei.items():
        if k.endswith("_function"):
            # map function name to actual Python function
            input_vars[k] = FUNCTION_REGISTRY[v]
        else:
            input_vars[k] = v

    def k_exp(soc): # TODO: Sieso sartukoziou csurf-ekiko cave beharrean
        x_0 = input_vars["x_0"]
        x_100 = input_vars["x_100"]
        sol_n = soc * (x_100 - x_0) + x_0
        eta = input_vars["U_n"](sol_n) - input_vars["U_sei"]
        return np.exp(
            -input_vars["alpha"] * input_vars["F"] * eta /
            (input_vars["R"] * input_vars["T"])
        ) * input_vars["k_sei"]

    input_vars["k_exp"] = k_exp

# ---------------- Plating ----------------
def _load_plating(cfg, input_vars):
    plating = cfg["Plating"]
    for k, v in plating.items():
        if k.endswith("_function"):
            # map function name to actual Python function
            input_vars[k] = FUNCTION_REGISTRY[v]
        else:
            input_vars[k] = v

# LEGACY

def cast_params_for_LAM(param, input_vars):
    A = param["Electrode height [m]"] * param["Electrode width [m]"]
    F = 96485.33212
    input_vars["A"] = A
    input_vars["F"] = F

    names = {"Positive": "p", "Negative": "n"}
    for k, domain in names.items():
        input_vars["Omega_"+domain] = param[k + " electrode partial molar volume [m3.mol-1]"]
        input_vars["omega_fact_" + domain] = 3
        input_vars["E_"+domain] = param[k + " electrode Young\'s modulus [Pa]"]
        input_vars["nu_"+domain] = param[k + " electrode Poisson\'s ratio"]
        input_vars["stress_critical_"+domain] = param[k + " electrode critical stress [Pa]"]
        input_vars["Beta_"+domain] = param[k + " electrode LAM constant proportional term [s-1]"]
        input_vars["m_"+domain] = param[k + " electrode LAM constant exponential term"]

        D = param[k + " electrode diffusivity [m2.s-1]"]
        R = param[k + " particle radius [m]"]
        L = param[k + " electrode thickness [m]"]

        #input_vars["Rpart_"+domain] = R

        def make_cs_cave(domain):
            def cs_cave(I, eps):
                as_particle = eps * 3 / input_vars["R_" + domain]
                I = -I
                if domain == "n":
                    j = I / (input_vars["F"] * as_particle * input_vars["L_"+domain] * input_vars["A"])
                else:
                    j = -I / (input_vars["F"] * as_particle * input_vars["L_"+domain] * input_vars["A"])
                return -j / 5 * input_vars["R_" + domain] / input_vars["D_" + domain]
            return cs_cave

        input_vars["cs_cave_"+domain] = make_cs_cave(domain)
        input_vars["L_" + domain] = L
        input_vars["D_" + domain] = D
        input_vars["R_" + domain] = R

        input_vars["csmax_"+domain] = param["Maximum concentration in " + k.lower() + " electrode [mol.m-3]"]
        input_vars["cini_"+domain] = param["Initial concentration in " + k.lower() + " electrode [mol.m-3]"]
        input_vars["cini_" + domain + "_0"] = param["Initial concentration in " + k.lower() + " electrode [mol.m-3]"]
        input_vars["U_"+domain] = param[k + " electrode OCP [V]"]
    input_vars["V_max"] = param["Upper voltage cut-off [V]"]
    input_vars["V_min"] = param["Lower voltage cut-off [V]"]
    input_vars["Q_nom"] = param["Nominal cell capacity [A.h]"]

    return input_vars

def cast_params_for_SEI(param, input_vars):

    k_sei = float(param.get("SEI kinetic rate constant [m.s-1]", 1e-12))
    D_ec = float(param.get("EC diffusivity [m2.s-1]", 2e-18))  # or "Molar mass of SEI"
    c_ec = float(param.get("EC initial concentration in electrolyte [mol.m-3]", 4541))
    alpha = float(param.get("SEI charge transfer coefficient", 0.5))
    U_sei = float(param.get("SEI open-circuit potential [V]", 0.4))
    F = float(param.get("Faraday constant [C.mol-1]", 96485.33212))
    R = float(param.get("Ideal gas constant [J.mol-1.K-1]", 8.314462618))
    T = float(param.get("Ambient temperature [K]", 298.15))
    V_sei = float(param.get("SEI partial molar volume [m3.mol-1]", 9.585e-5))
    z = float(param.get("Ratio of lithium moles to SEI moles", 2.))
    rho_SEI = param["SEI resistivity [Ohm.m]"]

    def k_exp(soc):

        #sol_n = (soc - x_0) / (x_100 - x_0)
        x_0 = input_vars["x_0"]
        x_100 = input_vars["x_100"]
        sol_n = soc * (x_100 - x_0) + x_0

        #  = np.tile(SOL_n_cycle, n_cycles * 2)
        eta = input_vars["U_n"](sol_n) - input_vars["U_sei"]

        exp_vals = np.exp(-input_vars["alpha"] * input_vars["F"] * eta / (input_vars["R"] * input_vars["T"]))
        k_exp_val = exp_vals * input_vars["k_sei"]
        return k_exp_val

    A_ec = c_ec * V_sei / z

    input_vars["k_exp"] = k_exp
    input_vars["k_sei"] = k_sei
    input_vars["alpha"] = alpha
    input_vars["A_ec"] = A_ec
    input_vars["D_ec"] = D_ec
    input_vars["rho_SEI"] = rho_SEI
    input_vars["V_SEI"] = V_sei
    input_vars["z_SEI"] = z
    input_vars["U_sei"] = U_sei
    input_vars["F"] = F
    input_vars["R"] = R
    input_vars["T"] = T

    return input_vars


if __name__ == "__main__":
    import pybamm
    from utils.pybamm_submodels import update_LAM_params, StressBasedLAM_FP

    param_chen = pybamm.ParameterValues("Chen2020")
    param = pybamm.ParameterValues("OKane2022")
    param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
    param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
    # param["Ratio of lithium moles to SEI moles"] = 2.

    # param["EC diffusivity [m2.s-1]"] /= 1000
    # param["Positive electrode LAM constant proportional term [s-1]"] /= 100.
    # param["Negative electrode LAM constant proportional term [s-1]"] *= 200.
    # param["Positive electrode LAM constant proportional term [s-1]"] *= 5.
    # param["Negative electrode LAM constant proportional term [s-1]"] *= 1000.
    param["Upper voltage cut-off [V]"] = 4.2

    param = update_LAM_params(param)

    model = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"},
                                    build=False)
    model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")
    model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")
    model.build_model()

    write_all_params_to_yaml(param, model, "Chen2020_U_", "../input/M50_okane_modified_plating.yaml")

    loaded_param = load_params("../input/M50_okane_modified_plating.yaml")
    loaded_param["x_0"] = 0.
    loaded_param["x_100"] = 1.
    print(loaded_param)
    print(loaded_param["k_exp"](1.))

    # param_chen = pybamm.ParameterValues("Chen2020")
    # param_kane = pybamm.ParameterValues("OKane2022")
    # param = pybamm.ParameterValues("Mohtat2020")
    # param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
    # param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
    # param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
    # param["Electrolyte diffusivity [m2.s-1]"] = param["Electrolyte diffusivity [m2.s-1]"](1., 298.15)
    #
    # param_Ai = pybamm.ParameterValues("Ai2020")
    # parameters_to_change = ["Negative electrode partial molar volume [m3.mol-1]",
    #                         "Positive electrode partial molar volume [m3.mol-1]",
    #                         "Negative electrode LAM constant exponential term",
    #                         "Positive electrode LAM constant exponential term",
    #                         "Negative electrode LAM constant proportional term [s-1]",
    #                         "Positive electrode LAM constant proportional term [s-1]",
    #                         "Negative electrode critical stress [Pa]",
    #                         "Positive electrode critical stress [Pa]",
    #                         "Negative electrode Young's modulus [Pa]",
    #                         "Positive electrode Young's modulus [Pa]",
    #                         "Negative electrode Poisson's ratio",
    #                         "Positive electrode Poisson's ratio",
    #                         "Negative electrode reference concentration for free of deformation [mol.m-3]",
    #                         "Positive electrode reference concentration for free of deformation [mol.m-3]",
    #                         "Negative electrode volume change",
    #                         "Positive electrode volume change",
    #                         "Cell thermal expansion coefficient [m.K-1]"]
    # for p in parameters_to_change:
    #     param.update({p: param_Ai[p]}, check_already_exists=False)
    #
    # parameters_to_change = ["Lithium plating kinetic rate constant [m.s-1]"]
    # for p in parameters_to_change:
    #     param.update({p: param_kane[p]}, check_already_exists=False)
    #
    # param = update_LAM_params(param)
    #
    # model = pybamm.lithium_ion.SPMe(options={"SEI": "ec reaction limited", "x-average side reactions": "false"},
    #                                 build=False)
    # model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={
    #     "loss of active material": "stress-driven"}, x_average=True, phase="primary")
    # model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={
    #     "loss of active material": "stress-driven"}, x_average=True, phase="primary")
    # model.build_model()
    #
    # write_all_params_to_yaml(param, model, "Mohtat2020_U_","../input/Mohtat_modified_plating.yaml")
    #
    # loaded_param = load_params("../input/Mohtat_modified_plating.yaml")
    # print(loaded_param)