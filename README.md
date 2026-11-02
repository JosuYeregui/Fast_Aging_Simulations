# Fast Aging Simulations Tool

Code accompanying the paper *Accelerating physics-based battery aging simulations by orders of magnitude via analytic decoupling* (J. Yeregui, I. Lopetegi and U. Iraola, Mondragon Unibertsitatea; manuscript prepared for a Cell Press journal).

Under quasi-steady-state assumptions, the internal variables that drive degradation are computed algebraically from the cycling conditions. The rates of SEI growth, lithium plating and stress-driven loss of active material (negative and positive electrode) are then integrated piecewise within each cycle, without a numerical solver. Ageing is extrapolated across cycles with SciPy's adaptive `RK23`, where one right-hand-side evaluation is one simulated cycle.

## Installation

Python 3.12 is the version used for development.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists the versions the code was developed and tested with. Other versions have not been checked. `numpy` >= 2 is required (`utils/cycler.py` uses `np.trapezoid`), and `pybamm` is needed even for the example because `utils/` imports it.

## Usage

`evaluation.py` is the reference example. From the project root:

```
python evaluation.py
```

It loads the LG M50 parameters from `input/M50_okane_modified_plating.yaml` (the calibration multipliers are already included, so do not apply them again), defines a CC-CV charge followed by a CC discharge, runs `FastAging_Simulation` (`main.py`) until `tot_cycles` or until the capacity falls below `stop_Q` times its initial value, and plots SEI thickness, active material fractions, plating thickness and capacity against the cycle number.

To change the protocol, edit `steps` in `evaluation.py`. Besides `CCStep` and `CCCVStep`, `utils/cycler.py` provides `Rest` and `DynamicProfile`, which reads a driving profile from `Dynamic_Profiles/` (`HWFET`, `NEDC`, `NYCC`, `UDDS`, `US06`, `WLTC`). To use another cell, pass a different YAML file to `load_params`.

## Project structure

```
main.py              FastAging_Simulation: RK23 extrapolation across cycles
evaluation.py        Usage example
utils/               Cycler, OCV/capacity functions, degradation models, parameter loading, PyBaMM submodels
input/               Cell parameter set (YAML)
Dynamic_Profiles/    Driving profiles
tests/               Validation and fleet studies of the paper
legacy/              Earlier scripts, not used by the paper
```

## Tests and reproduction of the paper results

There is no unit-test suite. The tests of the paper are scripts under `tests/`:

| Paper section | Folder | Scripts |
| --- | --- | --- |
| Validation against physics-based ageing simulations (Table 1, Figures 2 and 3) | `tests/Validation/CC/` | `comparative_tests_solv_T.py` |
| Validation under dynamic driving cycles (Figure 4) | `tests/Validation/Dynamic/` | `dynamic_sulzer_v2.py`, `dynamic_ours_v3.py` |
| Fleet-scale simulation (Figure 5) | `tests/Fleet/` | `fleet_study.py`, `plot_fleet_results.py` |

Run each script from its own folder, because the input paths are relative. The scripts in `tests/Validation/` need the project root on the path: `PYTHONPATH=../../.. python <script>`. The scripts in `tests/Fleet/` set it themselves.

**Constant-current validation.** `comparative_tests_solv_T.py` ages the cell for 2,000 cycles with the full PyBaMM SPMe (baseline), the method of Sulzer et al. and the proposed framework. It reports execution times and relative errors and writes PDF figures to `tests/Validation/CC/Results/`. The PyBaMM parameter modifications are set inside the script.

**Dynamic validation.** A WLTC discharge followed by a CC-CV charge is repeated for up to 10,000 cycles, with the Sulzer method as reference (the full PyBaMM model is not feasible here).

1. `dynamic_sulzer_v2.py` runs the reference and saves `simulation_history_WLTC_sulzer_v2.pkl` (about 9.2 h of CPU time on our machine).
2. `dynamic_ours_v3.py` runs the proposed framework, loads that file and writes the comparison figures to `tests/Validation/Dynamic/Results/`.

The reference results for WLTC and UDDS are stored in the same folder, so step 2 can be run without step 1.

**Fleet study.** `fleet_study.py` crosses manufacturing samples, usage archetypes and constant temperatures, and ages each cell to 60% of its initial capacity (`eol_threshold`). The paper uses 3,200 cells (128 samples × 5 archetypes × 5 temperatures) with scenario `v4_low_plating_peak`.

```
cd tests/Fleet
python fleet_study.py --list
python fleet_study.py --scenario v4_low_plating_peak --n-jobs 12
python plot_fleet_results.py --dir fleet_results_v4_low_plating_peak
```

Results go to `fleet_results_<scenario>/` (`fleet_results.csv`, `manufacturing_samples.csv`, `scenario.json`, `figures/`). Interrupted runs resume unless `--no-resume` is given. The other `fleet_results_v*` folders are the outputs of the remaining scenarios defined in the script.

## Lead contact

Josu Yeregui (jyeregui@mondragon.edu).

## License

MIT License. See `LICENSE`.
