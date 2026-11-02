# Fast_Aging_Simulations
# Battery Degradation Simulation

## Description
A Python-based battery degradation / SEI growth simulator using a custom RK23 integrator. The repository contains the simulation core, parameter loaders, cycler profiles and an `evaluation.py` example that runs a long-term simulation and plots results.

## Requirements
1. Python 3.10+ 
2. Python packages: `numpy`, `scipy`, `matplotlib`, `pyyaml` (or install from `requirements.txt`)

## Installation
1. Create and activate a virtual environment:
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
2. Install dependencies:
   - `pip install -r requirements.txt`  
   or
   - `pip install numpy scipy matplotlib pyyaml`

## Usage
1. From the project root run the example evaluator:
   - `python evaluation.py`
2. Or open the project in PyCharm (`PyCharm 2025.2.3`), set the project interpreter to `.venv\Scripts\python.exe`, and run `evaluation.py`.

## Configuration
1. Edit simulation parameters in `evaluation.py` (for example `tot_cycles`, `initial_soc`).
2. Edit input parameters in `legacy/M50_okane_modified.yaml`.
3. Core entry points:
   - `main.py` provides `scipy_rk23_integrate`.
   - `utils/` contains `param_to_inputvars.py`, `cycler.py`, and `characterize.py`.

## Output

`evaluation.py` prints ODE/integration time and displays plots for: SEI thickness (`L`), active material volume fraction for both electrodes (`epsilon_n`, `epsilon_p`) and capacity (`Q`). Modify the script to save figures or export numerical history.

## Project structure (key files)
1. `evaluation.py`
2. `main.py`
3. `legacy/M50_okane_modified.yaml`
4. `utils/param_to_inputvars.py`
5. `utils/cycler.py`
6. `utils/characterize.py`

## Tests
- No tests included. Add unit tests under a `tests/` directory as needed.

## Contributing
- Fork, create a feature branch, and open a pull request.

## License
- See `LICENSE` in the repository (if present).