# demovuln

[![PyPI version](https://img.shields.io/pypi/v/demovuln.svg)](https://pypi.org/project/demovuln/)
[![Documentation Status](https://readthedocs.org/projects/demovuln/badge/?version=latest)](https://demovuln.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`demovuln` is a Python package for simulating temporally structured demographic perturbations in matrix population models and estimating integrated population vulnerability.

The package is designed for comparative demographic analyses in which perturbations differ in magnitude, duration, and recurrence. It provides tools to simulate individual perturbation trajectories, evaluate full perturbation grids, and compute an integrated vulnerability metric based on population reduction relative to an unperturbed baseline.

## Documentation

The full documentation, including API reference and example notebooks, is available at:

https://demovuln.readthedocs.io/en/latest/

## Installation

Install the latest released version from PyPI:

```bash
pip install demovuln
```

For local development:

```bash
git clone https://github.com/agimenezromero/demovuln.git
cd demovuln
python -m pip install -e ".[dev,docs]"
```

## Basic usage

```python
import numpy as np
from demovuln import MatrixPopulationModel, simulate_dynamics

A = np.array([
    [0.0, 2.0],
    [0.4, 0.7],
])

model = MatrixPopulationModel(A)

result = simulate_dynamics(
    model,
    target="adult_survival",
    magnitude=0.25,
    duration=1,
    period=3,
    t_max=50,
    recovery_steps=10,
)

print(result.reduction)
print(result.abundance)
```

## Perturbation-grid analysis

```python
import numpy as np
from demovuln import MatrixPopulationModel, PerturbationGrid, run_grid

A = np.array([
    [0.0, 2.0],
    [0.4, 0.7],
])

model = MatrixPopulationModel(A)

grid = PerturbationGrid(
    magnitudes=np.linspace(0, 1, 11),
    durations=[0, 1, 2, 3],
    periods=[1, 2, 3, 5, 10],
)

out = run_grid(
    model,
    target="adult_survival",
    grid=grid,
    t_max=50,
    recovery_steps=10,
)

print(out.vulnerability)
print(out.table.head())
```

## Demographic targets

The package supports perturbations to:

- `adult_survival`
- `juvenile_survival`
- `fecundity`
- `all`
- `custom`

By default, adult stages are inferred as source-stage columns with at least one fecundity entry, and juvenile stages are inferred as the remaining source-stage columns. These definitions can be specified explicitly:

```python
model = MatrixPopulationModel(
    A,
    adult_stages=[1],
    juvenile_stages=[0],
)
```

Custom perturbation targets can be defined with Boolean masks:

```python
custom_mask = np.array([
    [False, False],
    [True, False],
])

result = simulate_dynamics(
    model,
    target="custom",
    custom_mask=custom_mask,
    magnitude=0.5,
    duration=1,
    period=3,
    t_max=50,
)
```

## Conceptual summary

For a given perturbation regime, population reduction is computed as:

```text
rho = 100 * (1 - N_perturbed(T) / N_baseline(T))
```

where `N_perturbed(T)` is the final population size under perturbed dynamics and `N_baseline(T)` is the final population size under the unperturbed baseline.

Integrated vulnerability is the mean population reduction across the simulated perturbation space:

```text
Phi = mean(rho)
```

## Example notebooks

Example notebooks are available in the `notebooks/` directory and in the online documentation:

https://demovuln.readthedocs.io/en/latest/notebooks.html

## Development checks

Run:

```bash
pytest
python examples/basic_usage.py
ruff check demovuln tests examples
sphinx-build -W -E -b html docs docs/_build/html
```

## Citation

Citation metadata are provided in `CITATION.cff`.

## License

This package is distributed under the MIT License.
