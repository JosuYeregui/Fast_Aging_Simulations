"""
Fleet-scale aging simulation study crossing:
    - Manufacturing variability (parameter distributions)
    - Usage archetypes (commuter, delivery, rideshare, ...)
    - Climate (representative ambient temperatures)

Built around the user's existing framework:
    - load_params / input_vars dict
    - scipy_rk23_integrate
    - Cycler with CCStep / CCCVStep / DynamicProfile / Rest
    - Arrhenius-modulated SEI (temperature enters via scalar T in input_vars)

The single-cell runner is `run_single_simulation()` — treat it as the black box.
The sweep is driven by `run_fleet_study()`, configured by a named `Scenario`.

Usage (from this folder):
    python fleet_study.py --list
    python fleet_study.py --scenario v2_ref [--n-mfg 128] [--n-jobs 12]

Each scenario writes to fleet_results_<name>/. The legacy fleet_results/ folder
(first 300-sim run) is never written to.
"""

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc, norm

# utils/cycler.py uses np.trapezoid (numpy >= 2); keep numpy 1.x envs working
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

# --- User's framework ------------------------------------------------------
from main import FastAging_Simulation
from utils.param_to_inputvars import load_params
from utils.cycler import CCStep, CCCVStep, Cycler, DynamicProfile, Rest
from utils.characterize import get_OVC_func, set_c_ini, get_Qn, get_Qp, updated_model_Q

# Current calibration: the multipliers below are already baked into this file
PARAM_YAML = ROOT / "input" / "M50_okane_modified_plating.yaml"
# YAML used by the legacy run (5 May): BASE_MULTIPLIERS were applied on top of it
PARAM_YAML_LEGACY = ROOT / "input" / "M50_okane_modified_plating_old_newer.yaml"
LEGACY_DIR = HERE / "fleet_results"
MODES = ["LLI_SEI", "plating", "LAM_n", "LAM_p"]


# =============================================================================
# 1. USAGE ARCHETYPES
# =============================================================================
# Each archetype defines charge protocol (CC/CV) + discharge profile (dynamic
# or CC fallback). Driving profiles can be swapped for real traces later
# (emobpy, NREL FleetDNA). For now, archetypes are parameterized abstractly so
# the experiment is reproducible without external datasets.

@dataclass
class UsageArchetype:
    name: str
    charge_c_rate: float          # CC phase of CCCV charge
    charge_cutoff_c: float        # CV cutoff as fraction of 1C
    discharge_mean_c: float       # Mean discharge C-rate for dynamic profile
    discharge_std_c: float        # Std of discharge C-rate (dynamic variability)
    dod: float                    # Target depth of discharge (used if Scenario.honour_dod)
    v_high: float                 # Upper voltage
    v_low: float                  # Lower voltage
    cycles_per_day: float         # Sets the rest time per cycle (used if Scenario.rest)


ARCHETYPES = [
    UsageArchetype("commuter",       0.3, 0.05, 0.4, 0.15, 0.60, 4.20, 3.00, 1.0),
    UsageArchetype("delivery",       0.7, 0.10, 0.8, 0.25, 0.80, 4.20, 2.80, 1.5),
    UsageArchetype("rideshare",      1.5, 0.15, 1.0, 0.35, 0.75, 4.20, 2.80, 2.5),
    UsageArchetype("weekend_driver", 0.3, 0.05, 0.5, 0.20, 0.60, 4.15, 3.00, 0.3),
    UsageArchetype("fleet_vehicle",  0.5, 0.08, 0.6, 0.15, 0.85, 4.20, 2.80, 1.8),
]

# High-current usage: DC fast charging and aggressive driving
FAST_ARCHETYPES = [
    UsageArchetype("dc_fast",        2.0, 0.15, 0.6, 0.20, 0.70, 4.20, 2.80, 1.0),
    UsageArchetype("highway_fast",   3.0, 0.20, 1.0, 0.30, 0.80, 4.20, 2.80, 1.5),
    UsageArchetype("performance",    1.0, 0.10, 2.0, 0.60, 0.70, 4.20, 2.80, 0.7),
]


# =============================================================================
# 2. CLIMATE SCENARIOS
# =============================================================================
# Constant T per simulation is a simplification — in reality cells see a yearly
# cycle. For a first study, we use representative "effective" temperatures
# anchored to regional climate archetypes. This is defensible if you cite it
# as an annual-mean proxy and flag it as a limitation.

@dataclass
class ClimateScenario:
    name: str
    T_kelvin: float               # Effective ambient / cell temperature


CLIMATES = [
    ClimateScenario("cold",      273.15 +  5.0),
    ClimateScenario("temperate", 273.15 + 15.0),
    ClimateScenario("warm",     273.15 + 22.0),
    ClimateScenario("hot",     273.15 + 30.0),
]

CLIMATES_V2 = CLIMATES + [ClimateScenario("very_hot", 273.15 + 40.0)]


# =============================================================================
# 3. MANUFACTURING PARAMETER DISTRIBUTIONS
# =============================================================================
# Replace CVs with values defensible from teardown literature (Ecker 2015,
# Schmalstieg, etc.) before final submission.

@dataclass
class ParamSpec:
    key: str                      # Exact key as used in your input_vars dict
    cv: float                     # Coefficient of variation (std/mean)
    bounds: tuple = (None, None)  # Optional hard bounds


# V1: list used for the legacy run. "k_SEI" does not match the YAML key
# ("k_sei"), so SEI kinetics were silently not sampled. Kept as-is so that
# v1_baseline reproduces the legacy results.
MFG_PARAMS_V1 = [
    ParamSpec("eps_0_n",      0.02, (0.50, 0.85)),
    ParamSpec("eps_0_p",      0.02, (0.45, 0.80)),
    ParamSpec("L_0",          0.15, (1e-9, 15e-9)),
    ParamSpec("k_SEI",        0.20, (None, None)),
    ParamSpec("D_ec",         0.25, (None, None)),
    ParamSpec("Beta_n",       0.25, (None, None)),
    ParamSpec("Beta_p",       0.25, (None, None)),
    ParamSpec("R_int",        0.10, (None, None)),
    ParamSpec("Q",            0.02, (None, None)),
]

# V2: SEI kinetics key fixed, plating kinetics added
MFG_PARAMS_V2 = [
    ParamSpec("eps_0_n",      0.02, (0.50, 0.85)),
    ParamSpec("eps_0_p",      0.02, (0.45, 0.80)),
    ParamSpec("L_0",          0.15, (1e-9, 15e-9)),
    ParamSpec("k_sei",        0.20, (None, None)),
    ParamSpec("D_ec",         0.25, (None, None)),
    ParamSpec("Beta_n",       0.25, (None, None)),
    ParamSpec("Beta_p",       0.25, (None, None)),
    ParamSpec("k_plating",    0.30, (None, None)),
    ParamSpec("R_int",        0.10, (None, None)),
    ParamSpec("Q",            0.02, (None, None)),
]

MANUFACTURING_PARAMS = MFG_PARAMS_V1  # backwards-compatible name


# =============================================================================
# 4. SCENARIOS
# =============================================================================

# Calibration multipliers used by the legacy run on PARAM_YAML_LEGACY. PARAM_YAML
# already includes them, so v2 scenarios start from no multipliers.
BASE_MULTIPLIERS = {"D_ec": 1 / 150, "Beta_n": 20, "Beta_p": 1 / 2000, "k_plating": 1 / 100}


@dataclass
class Scenario:
    name: str
    description: str = ""
    param_yaml: str = str(PARAM_YAML)
    base_multipliers: dict = field(default_factory=dict)
    mfg_params: list = field(default_factory=lambda: list(MFG_PARAMS_V2))
    archetypes: list = field(default_factory=lambda: list(ARCHETYPES))
    climates: list = field(default_factory=lambda: list(CLIMATES_V2))
    honour_dod: bool = True       # Discharge only `dod` per cycle instead of down to v_low
    rest: bool = False            # Add idle time so that one cycle lasts 24/cycles_per_day h
    rest_after: str = "charge"    # Park after "charge" (high SOC) or "discharge" (scheduled charging)
    rest_scale: float = 1.0       # Fraction of the idle time simulated as rest (effective calendar time)
    E_act_Ds: float = 0.0         # Arrhenius activation energy of solid diffusivity [J/mol]; 0 = off
    n_manufacturing: int = 128
    max_cycles: int = 10000
    eol_threshold: float = 0.6
    seed: int = 42




SCENARIOS = {s.name: s for s in [
    Scenario("v1_baseline", "Legacy settings (reproduces fleet_results/), new metrics",
             param_yaml=str(PARAM_YAML_LEGACY), base_multipliers=dict(BASE_MULTIPLIERS),
             mfg_params=list(MFG_PARAMS_V1), climates=list(CLIMATES), honour_dod=False,
             n_manufacturing=15),
    Scenario("v2_ref", "k_sei/k_plating sampled, DoD honoured, +40 C climate"),
    Scenario("v2_calendar", "v2_ref + rest periods (calendar ageing at parked SOC)", rest=True),
    Scenario("v2_fastcharge", "v2_ref + DC fast charge / performance archetypes",
             archetypes=list(ARCHETYPES) + list(FAST_ARCHETYPES)),
    Scenario("v2_lam_prone", "Crack-prone chemistry: Beta_n x5, Beta_p x5",
             base_multipliers={"Beta_n": 5, "Beta_p": 5}),
    Scenario("v2_low_plating", "Plating-resistant anode: k_plating x0.1",
             base_multipliers={"k_plating": 0.1}),
    # Tuned on 16 mfg samples: same mode mix as v2_low_plating, ~2x the SEI share
    # (mostly low-usage archetypes and 40 C) and life varying ~2x with temperature
    Scenario("v3_low_plating_T",
             "v2_low_plating + T-dependent LAM (E_act_Ds), stronger SEI/plating Arrhenius, short rest",
             base_multipliers={"k_plating": 0.1, "E_act_SEI": 1.3, "E_act_j0": 1.3, "D_ec": 0.7},
             E_act_Ds=8000, rest=True, rest_after="discharge", rest_scale=0.03),
    # Tuned (32 and 128 mfg samples) so that median life peaks at 25-30 C: steeper SEI Arrhenius
    # (2.2 x 38 kJ/mol) penalises the hot end, a weaker E_act_Ds softens the cold end
    Scenario("v4_low_plating_peak",
             "v3_low_plating_T with life peaking at 25-30 C: E_act_SEI x2.2, E_act_Ds 3 kJ/mol",
             base_multipliers={"k_plating": 0.1, "E_act_SEI": 2.2, "E_act_j0": 1.3},
             E_act_Ds=3000, rest=True, rest_after="discharge", rest_scale=0.03),
]}


def load_base_input_vars(base_multipliers: dict, param_yaml=PARAM_YAML) -> dict:
    base_input_vars = load_params(str(param_yaml))
    for k, m in base_multipliers.items():
        base_input_vars[k] *= m
    return base_input_vars


# =============================================================================
# 5. SAMPLING
# =============================================================================

def sample_manufacturing(n_samples: int, nominal_vars: dict, seed: int = 0,
                         mfg_params: list = None) -> pd.DataFrame:
    """
    Sobol-sampled log-normal perturbations around the nominal values from
    the YAML file. Reuses the same sample for every archetype/climate combo
    so manufacturing effects can be isolated cleanly.
    """
    mfg_params = mfg_params or MANUFACTURING_PARAMS
    specs = [s for s in mfg_params if s.key in nominal_vars]
    missing = [s.key for s in mfg_params if s.key not in nominal_vars]
    if missing:
        print(f"[warn] Skipping params not in input_vars: {missing}")

    sampler = qmc.Sobol(d=len(specs), scramble=True, seed=seed)
    u = sampler.random(n_samples)  # [0, 1]^d

    cols = {}
    for i, spec in enumerate(specs):
        mean = float(nominal_vars[spec.key])
        sigma_ln = np.sqrt(np.log(1 + spec.cv ** 2))
        mu_ln = np.log(abs(mean)) - 0.5 * sigma_ln ** 2
        z = norm.ppf(np.clip(u[:, i], 1e-6, 1 - 1e-6))
        vals = np.sign(mean) * np.exp(mu_ln + sigma_ln * z)

        lo, hi = spec.bounds
        if lo is not None:
            vals = np.maximum(vals, lo)
        if hi is not None:
            vals = np.minimum(vals, hi)
        cols[spec.key] = vals

    return pd.DataFrame(cols)


# =============================================================================
# 6. CYCLER CONSTRUCTION
# =============================================================================

def build_cycler(
    archetype: UsageArchetype,
    input_vars: dict,
    initial_soc: float,
    dynamic_rng: np.random.Generator,
    dynamic_duration_s: float = 1800.0,
    dynamic_dt_s: float = 10.0,
    honour_dod: bool = False,
    rest: bool = False,
    rest_after: str = "charge",
    rest_scale: float = 1.0,
) -> Cycler:
    """
    Mixed protocol: CCCV charge + dynamic discharge.

    Dynamic discharge is a random-walk current trace centered on the
    archetype's mean discharge C-rate.

    honour_dod=False: the trace is repeated until v_low (legacy behaviour).
    honour_dod=True:  the trace lasts dod / discharge_mean_c hours and is played
                      once; the cycle starts at SOC = 1 - dod so it is periodic.
    rest=True:        a rest step pads the cycle to 24 / cycles_per_day hours. It
                      follows the charge (rest_after="charge", parked at high SOC)
                      or the discharge (rest_after="discharge", parked at 1 - dod
                      and charged just before driving). rest_scale < 1 keeps
                      only that fraction of the idle time.
    """
    # --- Dynamic discharge trace ------------------------------------------
    if honour_dod:
        dynamic_duration_s = archetype.dod / archetype.discharge_mean_c * 3600
    n_steps = int(dynamic_duration_s / dynamic_dt_s)
    noise = dynamic_rng.normal(0.0, archetype.discharge_std_c, n_steps)
    smooth = np.convolve(noise, np.ones(5) / 5, mode="same")
    c_profile = archetype.discharge_mean_c + smooth
    c_profile = np.clip(c_profile, 0.05, 3.0)
    # Sign convention: discharge negative (matches CCStep usage)
    current_profile = -c_profile * input_vars["Q"]
    time_profile = np.arange(n_steps) * dynamic_dt_s / 3600

    discharge_step = DynamicProfile(
        t_vec=time_profile,
        I_vec=current_profile,
        V_lim=[archetype.v_low, 4.5],
        cycle=not honour_dod
    )

    charge_step = CCCVStep(
        I=archetype.charge_c_rate * input_vars["Q"],
        V_end=archetype.v_high,
        Cutoff_I=archetype.charge_cutoff_c * input_vars["Q"],
        Vlim=[archetype.v_low, archetype.v_high],
    )

    if honour_dod:
        initial_soc = 1.0 - archetype.dod

    OCV_fn = get_OVC_func(input_vars)
    cycler = Cycler(
        steps=[charge_step, discharge_step],
        SOC0=initial_soc,
        R=input_vars["R_int"],
        Q=input_vars["Q"],
        OCV_fn=OCV_fn,
    )
    if not rest:
        return cycler

    # Cycler times are in hours
    t_rest = rest_scale * max(24.0 / archetype.cycles_per_day - cycler.total_time, 0.0)
    if rest_after == "charge":
        steps = [charge_step, Rest(t_rest), discharge_step]
    elif rest_after == "discharge":
        steps = [charge_step, discharge_step, Rest(t_rest)]
    else:
        raise ValueError(f"rest_after must be 'charge' or 'discharge', got {rest_after!r}")
    return Cycler(
        steps=steps,
        SOC0=initial_soc,
        R=input_vars["R_int"],
        Q=input_vars["Q"],
        OCV_fn=OCV_fn,
    )


# =============================================================================
# 7. DEGRADATION-MODE ATTRIBUTION
# =============================================================================

def lli_ah(input_vars: dict, y) -> tuple:
    """Lithium inventory lost to SEI and plating [Ah], as in updated_model_Q."""
    F = 96485.33
    a_s = y[1] * 3 / input_vars["R_n"]
    geo = input_vars["A"] * F / 3600. * a_s * input_vars["L_n"]
    lli_sei = (y[0] - input_vars["L_0"]) * geo / input_vars["V_SEI"] * input_vars["z_SEI"]
    lli_pl = y[3] * geo / input_vars["V_Li"]
    return lli_sei, lli_pl


def attribute_capacity_loss(input_vars0: dict, cycler0: Cycler, y0, y_end) -> dict:
    """
    Split the capacity fade into mechanisms.

    input_vars0 / cycler0 must be the *initial* state (after set_c_ini). Both are
    copied, never modified.

    Returns:
        Ah inventory: LLI_SEI_Ah, LLI_pl_Ah, LAM_n_Ah, LAM_p_Ah (lithium / electrode
            capacity lost, not necessarily cell capacity).
        Counterfactual cell-capacity loss: capacity from the initial state with only
            one mechanism applied (dQ_LLI, dQ_LAM_n, dQ_LAM_p). dQ_LLI is split into
            dQ_LLI_SEI / dQ_plating proportionally to their Ah. dQ_residual is the
            interaction term: (Q_ref - Q_all) - sum of the individual losses.
        dominant_mode: argmax of the counterfactual losses.
    """
    y0 = np.asarray(y0, dtype=float)
    y_end = np.asarray(y_end, dtype=float)

    lli_sei, lli_pl = lli_ah(input_vars0, y_end)
    out = {
        "LLI_SEI_Ah": float(lli_sei),
        "LLI_pl_Ah": float(lli_pl),
        "LAM_n_Ah": float(get_Qn(input_vars0, y0[1]) - get_Qn(input_vars0, y_end[1])),
        "LAM_p_Ah": float(get_Qp(input_vars0, y0[2]) - get_Qp(input_vars0, y_end[2])),
    }

    def q_at(y):
        try:
            return float(updated_model_Q(y, copy.deepcopy(input_vars0), copy.deepcopy(cycler0),
                                         y0[0], y0[3]))
        except Exception:
            return np.nan

    y_lli = y0.copy();  y_lli[[0, 3]] = y_end[[0, 3]]
    y_lamn = y0.copy(); y_lamn[1] = y_end[1]
    y_lamp = y0.copy(); y_lamp[2] = y_end[2]

    Q_ref = q_at(y0)
    dQ_lli = Q_ref - q_at(y_lli)
    dQ_lamn = Q_ref - q_at(y_lamn)
    dQ_lamp = Q_ref - q_at(y_lamp)
    dQ_all = Q_ref - q_at(y_end)

    lli_tot = lli_sei + lli_pl
    f_sei = lli_sei / lli_tot if lli_tot > 0 else 0.5
    out.update({
        "dQ_LLI_SEI": dQ_lli * f_sei,
        "dQ_plating": dQ_lli * (1 - f_sei),
        "dQ_LAM_n": dQ_lamn,
        "dQ_LAM_p": dQ_lamp,
        "dQ_all": dQ_all,
        "dQ_residual": dQ_all - (dQ_lli + dQ_lamn + dQ_lamp),
    })
    losses = {m: out[f"dQ_{m}"] for m in MODES}
    if all(np.isnan(v) for v in losses.values()):
        out["dominant_mode"] = None
    else:
        out["dominant_mode"] = max(losses, key=lambda m: -np.inf if np.isnan(losses[m]) else losses[m])
    return out


def legacy_dominant_mode(y0, y_end) -> str:
    """Old label: largest relative change. Biased towards SEI (L_SEI grows ~60x)."""
    d_LSEI = (y_end[0] - y0[0]) / max(y0[0], 1e-12)
    d_eps_n = (y0[1] - y_end[1]) / max(y0[1], 1e-12)
    d_eps_p = (y0[2] - y_end[2]) / max(y0[2], 1e-12)
    d_plat = y_end[3] / max(y0[0], 1e-12)
    modes = {"LLI_SEI": d_LSEI, "LAM_n": d_eps_n, "LAM_p": d_eps_p, "plating": d_plat}
    return max(modes, key=modes.get)


# =============================================================================
# 8. SINGLE-SIMULATION RUNNER (black-box wrapper)
# =============================================================================

def run_single_simulation(
    manufacturing: dict,
    archetype: UsageArchetype,
    climate: ClimateScenario,
    base_input_vars: dict,
    max_cycles: int = 3000,
    eol_threshold: float = 0.6,
    initial_soc: float = 0.1,
    seed: int = 0,
    honour_dod: bool = False,
    rest: bool = False,
    rest_after: str = "charge",
    rest_scale: float = 1.0,
    E_act_Ds: float = 0.0,
) -> dict:
    """
    Run one cell through its full life. Returns a flat dict of summary
    metrics suitable for tabular aggregation.
    """
    # --- Compose the per-sim input_vars -----------------------------------
    input_vars = copy.deepcopy(base_input_vars)
    for k, v in manufacturing.items():
        input_vars[k] = float(v)
    input_vars["T"] = float(climate.T_kelvin)  # <-- Arrhenius entry point
    if E_act_Ds:
        # cs_cave ~ 1/D_s: slower diffusion when cold -> larger gradient -> more stress (LAM)
        # and a more extreme surface concentration (plating). D_s(T) = D_s,ref * exp(-Ea/R (1/T - 1/298.15))
        factor = np.exp(E_act_Ds / input_vars["R"] * (1 / input_vars["T"] - 1 / 298.15))
        for d in ("n", "p"):
            input_vars[f"cs_cave_{d}"] = lambda I, eps, f=input_vars[f"cs_cave_{d}"]: factor * f(I, eps)

    # --- Build cycler -----------------------------------------------------
    rng = np.random.default_rng(seed)
    cycler = build_cycler(archetype, input_vars, initial_soc, rng,
                          honour_dod=honour_dod, rest=rest, rest_after=rest_after,
                          rest_scale=rest_scale)
    input_vars = set_c_ini(cycler, input_vars)
    cycle_h = float(cycler.total_time)

    # --- Integrate --------------------------------------------------------
    y0_vec = [
        input_vars["L_0"],
        input_vars["eps_0_n"],
        input_vars["eps_0_p"],
        0.0,  # plating
    ]

    stop_Q = eol_threshold
    t0 = time.perf_counter()
    try:
        # FastAging_Simulation deep-copies input_vars and cycler, so both stay
        # at their initial state (needed by attribute_capacity_loss)
        history = FastAging_Simulation(
            y0_vec=y0_vec,
            input_vars=input_vars,
            tot_cycles=max_cycles,
            cycler=cycler,
            stop_Q=stop_Q,
            rtol=[1e-3, 1e-3, 1e-3, 1e-3],
            atol=[1e-9, 1e-4, 1e-4, 1e-9],
            stop_condition_fn=None,
            max_step=None,
            print_Qmin=False,
        )
        success = True
        err = None
    except Exception as e:
        success = False
        err = str(e)
        history = None
    runtime = time.perf_counter() - t0

    # --- Extract summary --------------------------------------------------
    if not success or history is None or len(history["n_cycles"]) == 0:
        return {
            "cycles_to_eol": None,
            "reached_eol": None,
            "truncated": None,
            "cycle_duration_h": cycle_h,
            "years_to_eol": None,
            "final_Q": None,
            "final_Q_fraction": None,
            "final_L_SEI": None,
            "final_eps_n": None,
            "final_eps_p": None,
            "final_plating": None,
            "plating_onset_cycle": None,
            "dominant_mode": None,
            "dominant_mode_legacy": None,
            "runtime_s": runtime,
            "success": False,
            "error": err,
        }

    n_cycles = np.asarray(history["n_cycles"])
    Q_hist = np.asarray(history["Q"])
    y_hist = np.asarray(history["y"])
    L_SEI = y_hist[:, 0]
    eps_n = y_hist[:, 1]
    eps_p = y_hist[:, 2]
    plating = y_hist[:, 3]

    # End-of-life = first cycle where Q drops below threshold (if any)
    Q_frac = Q_hist / Q_hist[0]
    below = np.where(Q_frac < eol_threshold)[0]
    cycles_to_eol = int(n_cycles[below[0]]) if below.size else int(n_cycles[-1])
    reached_eol = bool(below.size or Q_frac[-1] < eol_threshold + 0.01)
    # FastAging_Simulation stops silently (bare except) if a cycler step raises,
    # e.g. the DynamicProfile index error once the trace hits v_low on its last sample
    truncated = not reached_eol and n_cycles[-1] < max_cycles - 1

    # Plating onset: first point where plated lithium exceeds 1% of Q0
    _, lli_pl_hist = lli_ah(input_vars, y_hist.T)
    plating_above = np.where(lli_pl_hist > 0.01 * Q_hist[0])[0]
    plating_onset = int(n_cycles[plating_above[0]]) if plating_above.size else None

    attribution = attribute_capacity_loss(input_vars, cycler, y_hist[0], y_hist[-1])

    return {
        "cycles_to_eol": cycles_to_eol,
        "reached_eol": reached_eol,
        "truncated": truncated,
        "cycle_duration_h": cycle_h,
        "years_to_eol": cycles_to_eol * cycle_h / 8760.,
        "final_Q": float(Q_hist[-1]),
        "final_Q_fraction": float(Q_frac[-1]),
        "final_L_SEI": float(L_SEI[-1]),
        "final_eps_n": float(eps_n[-1]),
        "final_eps_p": float(eps_p[-1]),
        "final_plating": float(plating[-1]),
        "plating_onset_cycle": plating_onset,
        **attribution,
        "dominant_mode_legacy": legacy_dominant_mode(y_hist[0], y_hist[-1]),
        "runtime_s": runtime,
        "success": True,
        "error": None,
    }


# =============================================================================
# 9. PARALLEL EXECUTION
# =============================================================================
# input_vars holds closures (cs_cave_*) that cannot be pickled, so each worker
# builds its own base_input_vars once and tasks only carry plain data.

_WORKER_BASE = {}


def _init_worker(base_multipliers, param_yaml):
    _WORKER_BASE["vars"] = load_base_input_vars(base_multipliers, param_yaml)


def _run_task(task):
    key, kwargs = task
    multipliers = kwargs.pop("extra_multipliers", None)
    base = _WORKER_BASE["vars"]
    if multipliers:
        base = copy.copy(base)
        for k, m in multipliers.items():
            base[k] = base[k] * m
    return key, run_single_simulation(base_input_vars=base, **kwargs)


def run_tasks(tasks, base_multipliers, n_jobs=1, on_result=None, param_yaml=PARAM_YAML):
    """
    Run (key, kwargs) tasks, kwargs going to run_single_simulation (without
    base_input_vars; optional 'extra_multipliers' on top of base_multipliers).
    Calls on_result(key, result) as results arrive, in order.
    """
    if n_jobs == 1:
        _init_worker(base_multipliers, param_yaml)
        for task in tasks:
            on_result(*_run_task(task))
        return
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                             initargs=(base_multipliers, param_yaml)) as ex:
        for key, result in ex.map(_run_task, tasks, chunksize=8):
            on_result(key, result)


# =============================================================================
# 10. FLEET LOOP
# =============================================================================

def run_fleet_study(
    scenario: Scenario,
    output_dir: str = None,
    n_manufacturing: int = None,
    n_jobs: int = 1,
    log_every: int = 100,
    checkpoint_every: int = 1000,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Full crossed fleet study for one scenario.

    Total simulations = n_manufacturing * len(archetypes) * len(climates).
    Writes incremental checkpoints every `checkpoint_every` sims — long runs
    are resumable by re-calling with resume=True.
    """
    out = Path(output_dir) if output_dir else HERE / f"fleet_results_{scenario.name}"
    if out.resolve() == LEGACY_DIR.resolve():
        raise ValueError(f"{LEGACY_DIR} holds the legacy results and is read-only")
    out.mkdir(parents=True, exist_ok=True)
    results_csv = out / "fleet_results.csv"
    mfg_csv = out / "manufacturing_samples.csv"
    n_manufacturing = n_manufacturing or scenario.n_manufacturing

    with open(out / "scenario.json", "w") as f:
        json.dump({**asdict(scenario), "n_manufacturing": n_manufacturing}, f, indent=2, default=str)

    # --- Load nominal parameters -----------------------------------------
    base_input_vars = load_base_input_vars(scenario.base_multipliers, scenario.param_yaml)

    # --- Manufacturing samples (cached) ----------------------------------
    if resume and mfg_csv.exists():
        mfg_df = pd.read_csv(mfg_csv)
        print(f"[resume] Loaded {len(mfg_df)} cached manufacturing samples")
    else:
        mfg_df = sample_manufacturing(n_manufacturing, base_input_vars, seed=scenario.seed,
                                      mfg_params=scenario.mfg_params)
        mfg_df.to_csv(mfg_csv, index=False)

    # --- Previous results (for resume) -----------------------------------
    done_keys = set()
    if resume and results_csv.exists():
        prev = pd.read_csv(results_csv)
        done_keys = set(zip(prev["archetype"], prev["climate"], prev["mfg_idx"]))
        print(f"[resume] Skipping {len(done_keys)} completed simulations")
        all_results = prev.to_dict("records")
    else:
        all_results = []

    total = len(scenario.archetypes) * len(scenario.climates) * len(mfg_df)
    lookup = {}
    tasks = []
    for archetype in scenario.archetypes:
        for climate in scenario.climates:
            for idx in range(len(mfg_df)):
                key = (archetype.name, climate.name, idx)
                if key in done_keys:
                    continue
                mfg = mfg_df.iloc[idx].to_dict()
                lookup[key] = (archetype, climate, mfg)
                tasks.append((key, dict(
                    manufacturing=mfg, archetype=archetype, climate=climate,
                    max_cycles=scenario.max_cycles, eol_threshold=scenario.eol_threshold,
                    seed=scenario.seed + idx, honour_dod=scenario.honour_dod, rest=scenario.rest,
                    rest_after=scenario.rest_after, rest_scale=scenario.rest_scale,
                    E_act_Ds=scenario.E_act_Ds,
                )))

    done = len(done_keys)
    t_start = time.perf_counter()

    def on_result(key, result):
        nonlocal done
        archetype, climate, mfg = lookup[key]
        all_results.append({
            "archetype": archetype.name,
            "climate": climate.name,
            "T_kelvin": climate.T_kelvin,
            "mfg_idx": key[2],
            **{f"mfg_{k}": v for k, v in mfg.items()},
            **result,
        })
        done += 1

        if done % log_every == 0:
            elapsed = time.perf_counter() - t_start
            rate = (done - len(done_keys)) / max(elapsed, 1e-6)
            eta = (total - done) / max(rate, 1e-6)
            print(f"  {done}/{total}  rate={rate:.1f} sim/s  ETA={eta/60:.1f} min")

        if done % checkpoint_every == 0:
            pd.DataFrame(all_results).to_csv(results_csv, index=False)

    print(f"[{scenario.name}] {scenario.description}: {len(tasks)} sims to run")
    run_tasks(tasks, scenario.base_multipliers, n_jobs=n_jobs, on_result=on_result,
              param_yaml=scenario.param_yaml)

    df = pd.DataFrame(all_results)
    df.to_csv(results_csv, index=False)
    print(f"\nTotal runtime: {(time.perf_counter() - t_start)/60:.1f} min")
    print(f"Results: {results_csv}")
    return df


# =============================================================================
# 11. ANALYSIS
# =============================================================================

def drop_truncated(df: pd.DataFrame) -> pd.DataFrame:
    """Remove runs that stopped early because of a solver/cycler exception."""
    if "truncated" in df.columns:
        mask = df["truncated"].fillna(False).astype(bool)
    else:  # older CSVs: no run reaches max_cycles, so not reaching EOL means truncated
        mask = df["reached_eol"].eq(False)
    if mask.any():
        print(f"[warn] Excluding {mask.sum()} truncated runs ({mask.mean():.0%})")
    return df[~mask]


def summarize(df: pd.DataFrame, by=("archetype", "climate")) -> pd.DataFrame:
    """Headline stats per slice."""
    g = df.groupby(list(by))
    return g.agg(
        n=("success", "size"),
        success_rate=("success", "mean"),
        eol_p10=("cycles_to_eol", lambda x: x.quantile(0.10)),
        eol_p50=("cycles_to_eol", lambda x: x.quantile(0.50)),
        eol_p90=("cycles_to_eol", lambda x: x.quantile(0.90)),
        plating_rate=("plating_onset_cycle", lambda x: x.notna().mean()),
        mean_runtime=("runtime_s", "mean"),
    )


def sensitivity_spearman(df: pd.DataFrame, target: str = "cycles_to_eol") -> pd.DataFrame:
    """
    Per-slice Spearman rank correlation between each manufacturing parameter
    and the target. Use SALib for true Sobol indices in final analysis.
    """
    from scipy.stats import spearmanr
    mfg_cols = [c for c in df.columns if c.startswith("mfg_") and c != "mfg_idx"]
    rows = []
    for (archetype, climate), sub in df.groupby(["archetype", "climate"]):
        sub = sub.dropna(subset=[target])
        if len(sub) < 10:
            continue
        for col in mfg_cols:
            rho, p = spearmanr(sub[col], sub[target])
            rows.append({
                "archetype": archetype, "climate": climate,
                "param": col.replace("mfg_", ""), "rho": rho, "p": p,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 12. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=list(SCENARIOS))
    parser.add_argument("--list", action="store_true", help="List scenarios and exit")
    parser.add_argument("--n-mfg", type=int, default=None, help="Override n_manufacturing")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.list or args.scenario is None:
        for s in SCENARIOS.values():
            n = s.n_manufacturing * len(s.archetypes) * len(s.climates)
            print(f"{s.name:16s} {n:6d} sims  {s.description}")
        sys.exit(0)

    scenario = SCENARIOS[args.scenario]
    df = run_fleet_study(
        scenario,
        output_dir=args.output_dir,
        n_manufacturing=args.n_mfg,
        n_jobs=args.n_jobs,
        resume=not args.no_resume,
    )

    df = drop_truncated(df[df["success"].astype(bool)])
    print("\n=== Summary by archetype x climate ===")
    print(summarize(df))

    print("\n=== Dominant mode (capacity attribution) ===")
    print(df.groupby(["archetype", "climate"])["dominant_mode"].value_counts().unstack(fill_value=0))
