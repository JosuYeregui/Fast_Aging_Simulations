import pybamm
from matplotlib import pyplot as plt
import os
from copy import deepcopy
import time


from utils.characterize import *
from utils.pybamm_submodels import *

def get_Rint(model, param):
    experiment = pybamm.Experiment(["Charge at 1C until 4.2V", "Discharge at 1C until 2.5V"])
    sim = pybamm.Simulation(
        model,
        experiment=experiment,
        parameter_values=param
    )

    solution = sim.solve(initial_soc=0.)

    V = solution["Terminal voltage [V]"].data
    OCV = solution["Battery open-circuit voltage [V]"].data
    I = -solution["Current [A]"].data

    zero_current_mask = np.abs(I) < 1e-6

    V_pol = V - OCV

    R_int = np.full_like(I, np.nan)

    non_zero_indices = ~zero_current_mask
    R_int[non_zero_indices] = V_pol[non_zero_indices] / I[non_zero_indices]
    median_R_int = np.median(R_int)

    return np.abs(median_R_int)

if __name__ == "__main__":
    import copy
    initial_soc = 0.
    T_dev = -30.

    param_chen = pybamm.ParameterValues("Chen2020")
    param = pybamm.ParameterValues("OKane2022")
    param["Negative electrode OCP [V]"] = param_chen["Negative electrode OCP [V]"]
    param["Negative electrode diffusivity [m2.s-1]"] = param_chen["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode diffusivity [m2.s-1]"] = param_chen["Positive electrode diffusivity [m2.s-1]"]
    # param["Electrolyte diffusivity [m2.s-1]"] = param["Electrolyte diffusivity [m2.s-1]"](1., 298.15)
    func = copy.deepcopy(param["Electrolyte diffusivity [m2.s-1]"])
    param["Electrolyte diffusivity [m2.s-1]"] = lambda c, T: func(1., T)

    param["EC diffusivity [m2.s-1]"] /= 150
    param["Positive electrode LAM constant proportional term [s-1]"] /= 2000.
    param["Negative electrode LAM constant proportional term [s-1]"] *= 20.
    param["Lithium plating kinetic rate constant [m.s-1]"] /= 100.

    param["Initial temperature [K]"] += T_dev
    param["Ambient temperature [K]"] += T_dev

    param["Upper voltage cut-off [V]"] = 4.2
    param = update_LAM_params(param)
    param.set_initial_state(initial_soc)
    prev_params = deepcopy(param)

    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(param)

    model = pybamm.lithium_ion.SPMe(
        options={"SEI": "ec reaction limited", "x-average side reactions": "false",
                 "use lumped thermal capacity": "false"},
        build=False)

    model.submodels["Negative active material"] = StressBasedLAM(
        model.param, "Negative"
    )

    model.submodels["Negative active material"] = StressBasedLAM_FP(model.param, "Negative", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.submodels["Plating"] = Plating_FP(model.param, "negative", x_average=True,
                                            options={"lithium plating": "irreversible"})

    model.submodels["Positive active material"] = StressBasedLAM(
        model.param, "Positive"
    )

    model.submodels["Positive active material"] = StressBasedLAM_FP(model.param, "Positive", options={
        "loss of active material": "stress-driven"}, x_average=True, phase="primary")

    model.build_model()

    print(get_Rint(model, param))