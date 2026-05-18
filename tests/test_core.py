import numpy as np
import pytest

from demovuln import (
    MatrixPopulationModel,
    PerturbationGrid,
    simulate_dynamics,
    run_grid,
    population_reduction,
    compute_vulnerability,
)
from demovuln.perturbations import apply_perturbation


def simple_matrix():
    return np.array(
        [
            [0.0, 2.0],
            [0.4, 0.7],
        ]
    )


def test_model_accepts_valid_projection_matrix():
    model = MatrixPopulationModel(simple_matrix())
    assert model.n_stages == 2
    assert model.lambda_ > 0
    assert np.isclose(model.stable_distribution().sum(), 1.0)


def test_model_rejects_non_square_matrix():
    A = np.ones((2, 3))
    with pytest.raises(ValueError):
        MatrixPopulationModel(A)


def test_model_rejects_negative_entries():
    A = simple_matrix()
    A[0, 0] = -1.0
    with pytest.raises(ValueError):
        MatrixPopulationModel(A)


def test_zero_magnitude_gives_zero_reduction():
    model = MatrixPopulationModel(simple_matrix())

    sim = simulate_dynamics(
        model,
        target="adult_survival",
        magnitude=0.0,
        duration=1,
        period=2,
        t_max=20,
        recovery_steps=5,
    )

    assert abs(sim.reduction) < 1e-10


def test_zero_duration_gives_zero_reduction():
    model = MatrixPopulationModel(simple_matrix())

    sim = simulate_dynamics(
        model,
        target="adult_survival",
        magnitude=0.5,
        duration=0,
        period=2,
        t_max=20,
        recovery_steps=5,
    )

    assert abs(sim.reduction) < 1e-10


def test_positive_magnitude_reduces_population():
    model = MatrixPopulationModel(simple_matrix())

    sim = simulate_dynamics(
        model,
        target="adult_survival",
        magnitude=0.5,
        duration=1,
        period=2,
        t_max=20,
        recovery_steps=5,
    )

    assert sim.reduction > 0


def test_population_reduction_against_manual_calculation():
    reduction = population_reduction(
        final_population=75.0, baseline_final_population=100.0
    )
    assert np.isclose(reduction, 25.0)


def test_adult_survival_perturbation_scales_adult_column():
    A = simple_matrix()
    model = MatrixPopulationModel(A)

    B = apply_perturbation(model, target="adult_survival", magnitude=0.5)

    # By default, adults are columns with non-zero fecundity entries.
    # Here column 1 is adult because A[0, 1] > 0.
    expected = A.copy()
    expected[:, 1] *= 0.5

    assert np.allclose(B, expected)


def test_fecundity_perturbation_scales_only_fecundity_entries():
    A = simple_matrix()
    model = MatrixPopulationModel(A)

    B = apply_perturbation(model, target="fecundity", magnitude=0.5)

    expected = A.copy()
    expected[0, 1] *= 0.5

    assert np.allclose(B, expected)


def test_all_perturbation_double_scales_fecundity_by_default():
    A = simple_matrix()
    model = MatrixPopulationModel(A)

    B = apply_perturbation(model, target="all", magnitude=0.5)

    expected = A.copy()
    expected[0, 1] *= 0.25  # (1 - m)^2 for fecundity entries
    expected[1, 0] *= 0.5
    expected[1, 1] *= 0.5

    assert np.allclose(B, expected)


def test_simulation_returns_trajectory_with_expected_length():
    model = MatrixPopulationModel(simple_matrix())

    sim = simulate_dynamics(
        model,
        target="juvenile_survival",
        magnitude=0.25,
        duration=1,
        period=3,
        t_max=20,
        recovery_steps=5,
    )

    assert len(sim.abundance) == 26  # includes initial condition
    assert len(sim.baseline_abundance) == 26
    assert np.isfinite(sim.reduction)


def test_simulation_can_return_stage_vectors():
    model = MatrixPopulationModel(simple_matrix())

    sim = simulate_dynamics(
        model,
        target="juvenile_survival",
        magnitude=0.25,
        duration=1,
        period=3,
        t_max=20,
        recovery_steps=5,
        return_stage_vectors=True,
    )

    assert sim.stage_vectors.shape == (26, 2)
    assert sim.baseline_stage_vectors.shape == (26, 2)


def test_duration_cannot_exceed_period():
    model = MatrixPopulationModel(simple_matrix())

    with pytest.raises(ValueError):
        simulate_dynamics(
            model,
            target="adult_survival",
            magnitude=0.5,
            duration=5,
            period=2,
            t_max=20,
        )


def test_grid_vulnerability_is_mean_reduction():
    model = MatrixPopulationModel(simple_matrix())

    grid = PerturbationGrid(
        magnitudes=[0.0, 0.5],
        durations=[1],
        periods=[2, 3],
    )

    result = run_grid(
        model,
        target="adult_survival",
        grid=grid,
        t_max=20,
        recovery_steps=5,
    )

    assert len(result.table) == 4
    assert np.isclose(
        result.vulnerability,
        result.table["population_reduction"].mean(),
    )


def test_compute_vulnerability_from_table():
    model = MatrixPopulationModel(simple_matrix())

    grid = PerturbationGrid(
        magnitudes=[0.0, 0.5],
        durations=[1],
        periods=[2],
    )

    result = run_grid(
        model,
        target="adult_survival",
        grid=grid,
        t_max=20,
        recovery_steps=5,
    )

    phi = compute_vulnerability(result.table)
    assert np.isclose(phi, result.vulnerability)


def test_frequency_grid_conversion():
    grid = PerturbationGrid.from_frequencies(
        magnitudes=[0.1],
        durations=[1],
        frequencies=[1.0, 2.0],
        generation_time=4,
    )

    assert grid.periods == (2, 4)


def test_infeasible_grid_scenarios_can_be_kept_as_nan():
    model = MatrixPopulationModel(simple_matrix())

    grid = PerturbationGrid(
        magnitudes=[0.5],
        durations=[3],
        periods=[2],
    )

    result = run_grid(
        model,
        target="adult_survival",
        grid=grid,
        t_max=20,
        skip_infeasible=False,
    )

    assert len(result.table) == 1
    assert not bool(result.table.loc[0, "feasible"])
    assert np.isnan(result.table.loc[0, "population_reduction"])
