import numpy as np

from demovuln import MatrixPopulationModel, PerturbationGrid, run_grid, simulate_dynamics

# Simple stage-structured matrix: rows are destination stages, columns are source stages.
A = np.array(
    [
        [0.0, 2.0],
        [0.4, 0.7],
    ]
)

# With default settings, non-zero entries in the first row define fecundity, and
# columns with fecundity are treated as adult/reproductive stages.
model = MatrixPopulationModel(A, name="example_species")

# A single trajectory is directly accessible.
trajectory = simulate_dynamics(
    model,
    target="adult_survival",
    magnitude=0.2,
    duration=2,
    period=5,
    t_max=40,
    recovery_steps=10,
)
print("Population reduction for one scenario:", trajectory.reduction)
print("Abundance trajectory:", trajectory.abundance)

# A perturbation grid defines the perturbation space Ω.
grid = PerturbationGrid(
    magnitudes=np.linspace(0.0, 0.5, 6),
    durations=range(1, 5),
    periods=range(1, 9),
)

result = run_grid(
    model,
    target="adult_survival",
    grid=grid,
    t_max=40,
    recovery_steps=10,
)

print("Integrated vulnerability Φ:", result.vulnerability)
print(result.table.head())
