from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import MatrixPopulationModel, stable_stage_distribution
from .perturbations import Target, apply_perturbation


@dataclass(frozen=True)
class SimulationResult:
    """Result of a single temporally structured perturbation simulation.

    Attributes
    ----------
    abundance : numpy.ndarray
        Total population size through time under the perturbed dynamics. The
        first element corresponds to the initial state.
    baseline_abundance : numpy.ndarray
        Total population size through time under the unperturbed baseline
        dynamics, evaluated over the same time horizon.
    stage_vectors : numpy.ndarray or None
        Stage-vector trajectory under the perturbed dynamics. Returned only when
        ``return_stage_vectors=True``.
    baseline_stage_vectors : numpy.ndarray or None
        Stage-vector trajectory under the unperturbed baseline dynamics. Returned
        only when ``return_stage_vectors=True``.
    reduction : float
        Percent population reduction at the final time step, measured relative
        to the unperturbed baseline trajectory.
    final_population : float
        Final total population size under the perturbed dynamics.
    baseline_final_population : float
        Final total population size under the unperturbed baseline dynamics.
    magnitude : float
        Proportional reduction used in the perturbation.
    duration : int
        Number of consecutive projection intervals during which each
        perturbation event was active.
    period : int
        Number of projection intervals between consecutive perturbation onsets.
    target : str
        Demographic component affected by the perturbation.
    """

    abundance: np.ndarray
    baseline_abundance: np.ndarray
    stage_vectors: Optional[np.ndarray]
    baseline_stage_vectors: Optional[np.ndarray]
    reduction: float
    final_population: float
    baseline_final_population: float
    magnitude: float
    duration: int
    period: int
    target: str


def _validate_schedule(duration: int, period: int) -> tuple[int, int]:
    duration = int(duration)
    period = int(period)
    if duration < 0:
        raise ValueError("duration must be non-negative.")
    if period <= 0:
        raise ValueError("period must be positive.")
    if duration > period:
        raise ValueError("period must be greater than or equal to duration.")
    return duration, period


def _is_perturbed_step(t: int, *, start: int, duration: int, period: int) -> bool:
    if duration == 0 or t < start:
        return False
    return ((t - start) % period) < duration


def _project(
    A0: np.ndarray,
    A1: Optional[np.ndarray],
    n0: np.ndarray,
    *,
    t_max: int,
    recovery_steps: int,
    start: int,
    duration: int,
    period: int,
    force_during_recovery: bool,
    return_stage_vectors: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    total_steps = int(t_max) + int(recovery_steps)
    abundance = np.empty(total_steps + 1, dtype=float)
    vectors = np.empty((total_steps + 1, A0.shape[0]), dtype=float) if return_stage_vectors else None

    n = np.asarray(n0, dtype=float).copy()
    abundance[0] = np.sum(n)
    if vectors is not None:
        vectors[0, :] = n

    for t in range(total_steps):
        within_forcing_window = t < t_max or force_during_recovery
        use_perturbed = (
            A1 is not None
            and within_forcing_window
            and _is_perturbed_step(t, start=start, duration=duration, period=period)
        )
        A = A1 if use_perturbed else A0
        n = A @ n
        abundance[t + 1] = np.sum(n)
        if vectors is not None:
            vectors[t + 1, :] = n

    return abundance, vectors


def population_reduction(final_population: float, baseline_final_population: float) -> float:
    """Compute percent population reduction relative to a baseline trajectory.

    Parameters
    ----------
    final_population : float
        Final population size under the perturbed dynamics.
    baseline_final_population : float
        Final population size under the unperturbed baseline dynamics evaluated
        over the same time horizon.

    Returns
    -------
    float
        Percent population reduction, defined as
        ``100 * (1 - final_population / baseline_final_population)``.

    Raises
    ------
    ValueError
        If ``baseline_final_population`` is not positive.
    """
    if baseline_final_population <= 0:
        raise ValueError("baseline_final_population must be positive.")
    return 100.0 * (1.0 - float(final_population) / float(baseline_final_population))


def simulate_dynamics(
    model: MatrixPopulationModel | np.ndarray,
    *,
    target: Target = "adult_survival",
    magnitude: float,
    duration: int,
    period: int,
    t_max: int,
    recovery_steps: int = 0,
    start: int = 0,
    initial_state: Optional[np.ndarray] = None,
    normalize_by_lambda: bool = True,
    survival_affects_fecundity: bool = True,
    custom_mask: Optional[np.ndarray] = None,
    return_stage_vectors: bool = False,
    force_during_recovery: bool = False,
) -> SimulationResult:
    """Simulate a population trajectory under a demographic perturbation regime.

    Parameters
    ----------
    model : MatrixPopulationModel or array_like of shape (n_stages, n_stages)
        Unperturbed matrix population model. If a raw matrix is supplied, it is
        converted to ``MatrixPopulationModel`` using the default target
        definitions.
    target : {"adult_survival", "juvenile_survival", "fecundity", "all", "custom"}, default="adult_survival"
        Demographic component affected by the perturbation.
    magnitude : float
        Proportional reduction applied to the selected entries. Values must lie
        in the interval ``[0, 1]``.
    duration : int
        Number of consecutive projection intervals during which each
        perturbation event is active. A value of ``0`` produces no perturbation.
    period : int
        Number of projection intervals between the onset of consecutive
        perturbation events. Must be greater than or equal to ``duration``.
    t_max : int
        Number of projection intervals in the perturbation-forcing window.
    recovery_steps : int, default=0
        Number of additional unperturbed projection intervals after the forcing
        window.
    start : int, default=0
        Projection interval at which the first perturbation event starts.
    initial_state : array_like of shape (n_stages,), optional
        Initial population vector. If omitted, the stable stage distribution of
        the unperturbed model is used. If supplied, the vector is normalized to
        unit total abundance.
    normalize_by_lambda : bool, default=True
        If True, divide both the baseline and perturbed projection matrices by
        the dominant eigenvalue of the unperturbed matrix. This makes abundance
        trajectories relative to the asymptotic growth of the unperturbed model.
    survival_affects_fecundity : bool, default=True
        If True, survival perturbations scale whole source-stage columns,
        including fecundity entries. If False, fecundity entries are excluded
        from adult- and juvenile-survival perturbations.
    custom_mask : array_like of bool, optional
        Boolean matrix selecting entries to perturb when ``target="custom"``.
    return_stage_vectors : bool, default=False
        If True, store the full stage-vector trajectory for both perturbed and
        baseline dynamics.
    force_during_recovery : bool, default=False
        If True, scheduled perturbation events continue during the recovery
        window. If False, recovery steps use the unperturbed matrix.

    Returns
    -------
    SimulationResult
        Object containing abundance trajectories, optional stage-vector
        trajectories, final abundances and percent population reduction.

    Notes
    -----
    Population reduction is measured against the unperturbed baseline trajectory
    at the same final time step. This separates the effect of the perturbation
    regime from the intrinsic growth or decline of the unperturbed projection
    matrix.
    """
    if not isinstance(model, MatrixPopulationModel):
        model = MatrixPopulationModel(np.asarray(model, dtype=float))

    duration, period = _validate_schedule(duration, period)
    t_max = int(t_max)
    recovery_steps = int(recovery_steps)
    start = int(start)
    if t_max < 0 or recovery_steps < 0:
        raise ValueError("t_max and recovery_steps must be non-negative.")
    if start < 0:
        raise ValueError("start must be non-negative.")

    A0 = model.matrix
    A1 = apply_perturbation(
        model,
        target,
        magnitude,
        survival_affects_fecundity=survival_affects_fecundity,
        custom_mask=custom_mask,
    )

    if normalize_by_lambda:
        lam = model.lambda_
        A0 = A0 / lam
        A1 = A1 / lam

    if initial_state is None:
        n0 = stable_stage_distribution(model.matrix)
    else:
        n0 = np.asarray(initial_state, dtype=float)
        if n0.ndim != 1 or n0.shape[0] != model.n_stages:
            raise ValueError("initial_state must be a vector with one entry per stage.")
        total = np.sum(n0)
        if total <= 0:
            raise ValueError("initial_state must have positive total abundance.")
        n0 = n0 / total

    abundance, vectors = _project(
        A0,
        A1,
        n0,
        t_max=t_max,
        recovery_steps=recovery_steps,
        start=start,
        duration=duration,
        period=period,
        force_during_recovery=force_during_recovery,
        return_stage_vectors=return_stage_vectors,
    )
    baseline_abundance, baseline_vectors = _project(
        A0,
        None,
        n0,
        t_max=t_max,
        recovery_steps=recovery_steps,
        start=start,
        duration=0,
        period=period,
        force_during_recovery=False,
        return_stage_vectors=return_stage_vectors,
    )

    reduction = population_reduction(abundance[-1], baseline_abundance[-1])
    return SimulationResult(
        abundance=abundance,
        baseline_abundance=baseline_abundance,
        stage_vectors=vectors,
        baseline_stage_vectors=baseline_vectors,
        reduction=float(reduction),
        final_population=float(abundance[-1]),
        baseline_final_population=float(baseline_abundance[-1]),
        magnitude=float(magnitude),
        duration=int(duration),
        period=int(period),
        target=str(target),
    )
