from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


def _as_square_matrix(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square two-dimensional projection matrix.")
    if np.any(~np.isfinite(A)):
        raise ValueError("A contains NaN or infinite values.")
    if np.any(A < 0):
        raise ValueError("A must be non-negative for the demographic routines implemented here.")
    return A


def dominant_eigenvalue(A: np.ndarray) -> float:
    """Return the asymptotic growth rate of a projection matrix.

    Parameters
    ----------
    A : array_like of shape (n_stages, n_stages)
        Non-negative projection matrix. Columns represent source stages at one
        projection interval and rows represent destination stages at the next
        projection interval.

    Returns
    -------
    float
        Dominant eigenvalue of ``A``, corresponding to the asymptotic population
        growth rate under the unperturbed linear dynamics.

    Raises
    ------
    ValueError
        If ``A`` is not a finite, square, non-negative matrix or if the dominant
        eigenvalue is not positive.
    """
    A = _as_square_matrix(A)
    eigvals = np.linalg.eigvals(A)
    idx = int(np.argmax(np.real(eigvals)))
    value = float(np.real(eigvals[idx]))
    if value <= 0:
        raise ValueError("Dominant eigenvalue must be positive.")
    return value


def stable_stage_distribution(A: np.ndarray) -> np.ndarray:
    """Return the stable stage distribution of a projection matrix.

    Parameters
    ----------
    A : array_like of shape (n_stages, n_stages)
        Non-negative projection matrix.

    Returns
    -------
    numpy.ndarray of shape (n_stages,)
        Dominant right eigenvector of ``A``, normalized to sum to one.

    Notes
    -----
    For irreducible non-negative projection matrices, the normalized dominant
    right eigenvector gives the asymptotic distribution of individuals among
    stages. The returned vector is used as the default initial condition in
    simulation routines.
    """
    A = _as_square_matrix(A)
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(np.real(eigvals)))
    w = np.real(eigvecs[:, idx])

    # Eigenvectors may be returned with arbitrary sign. For primitive non-negative
    # matrices the dominant eigenvector has entries of common sign; abs is robust
    # for the empirical MPMs used in this framework.
    w = np.abs(w)
    total = float(np.sum(w))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Could not compute a valid stable stage distribution.")
    return w / total


def _indices_to_mask(indices: Optional[Iterable[int]], n: int, *, name: str) -> Optional[np.ndarray]:
    if indices is None:
        return None
    mask = np.zeros(n, dtype=bool)
    for i in indices:
        if i < 0 or i >= n:
            raise ValueError(f"{name} contains index {i}, outside [0, {n - 1}].")
        mask[int(i)] = True
    return mask


@dataclass(frozen=True)
class MatrixPopulationModel:
    """Matrix population model with explicit demographic target definitions.

    Parameters
    ----------
    A : array_like of shape (n_stages, n_stages)
        Projection matrix defining transitions and reproductive contributions
        over one projection interval. Columns are source stages at time ``t``;
        rows are destination stages at time ``t + 1``.
    fecundity_mask : array_like of bool, optional
        Boolean matrix identifying entries interpreted as fecundity or newborn
        production. If omitted, non-zero entries in ``fecundity_rows`` are used.
    fecundity_rows : tuple of int, default=(0,)
        Row indices corresponding to newborn or reproductive-output classes. By
        default, fecundity is inferred from non-zero entries in the first row.
    adult_stages : iterable of int, optional
        Column indices corresponding to source stages classified as adult or
        reproductive. If omitted together with ``juvenile_stages``, adult stages
        are inferred as columns contributing to at least one fecundity entry.
    juvenile_stages : iterable of int, optional
        Column indices corresponding to source stages classified as juvenile or
        pre-reproductive. If omitted, juvenile stages are inferred as all stages
        not classified as adult.
    name : str, optional
        Optional label for the model, population or species.
        
    Notes
    -----
    The projection matrix is stored internally as a NumPy array. The model also
    stores inferred or user-defined demographic masks identifying fecundity entries,
    adult source stages, juvenile source stages, and the dominant eigenvalue of the
    unperturbed projection matrix.

    Stage indices are zero-based, following Python indexing.
    """

    A: np.ndarray
    fecundity_mask: Optional[np.ndarray] = None
    fecundity_rows: tuple[int, ...] = (0,)
    adult_stages: Optional[Iterable[int]] = None
    juvenile_stages: Optional[Iterable[int]] = None
    name: Optional[str] = None
    _A: np.ndarray = field(init=False, repr=False)
    _fecundity_mask: np.ndarray = field(init=False, repr=False)
    _adult_stage_mask: np.ndarray = field(init=False, repr=False)
    _juvenile_stage_mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        A = _as_square_matrix(self.A)
        object.__setattr__(self, "_A", A.copy())
        n = A.shape[0]

        if self.fecundity_mask is None:
            rows = tuple(int(r) for r in self.fecundity_rows)
            if not rows:
                raise ValueError("At least one fecundity row must be supplied.")
            mask = np.zeros_like(A, dtype=bool)
            for row in rows:
                if row < 0 or row >= n:
                    raise ValueError(f"fecundity row {row} outside [0, {n - 1}].")
                mask[row, :] = A[row, :] > 0
        else:
            mask = np.asarray(self.fecundity_mask, dtype=bool)
            if mask.shape != A.shape:
                raise ValueError("fecundity_mask must have the same shape as A.")
        object.__setattr__(self, "_fecundity_mask", mask.copy())

        adult_mask = _indices_to_mask(self.adult_stages, n, name="adult_stages")
        juvenile_mask = _indices_to_mask(self.juvenile_stages, n, name="juvenile_stages")

        if adult_mask is None and juvenile_mask is None:
            adult_mask = np.any(mask, axis=0)
            juvenile_mask = ~adult_mask
        elif adult_mask is None:
            adult_mask = ~juvenile_mask
        elif juvenile_mask is None:
            juvenile_mask = ~adult_mask

        if np.any(adult_mask & juvenile_mask):
            raise ValueError("adult_stages and juvenile_stages overlap.")
        if not np.any(adult_mask):
            raise ValueError("No adult stages were identified.")
        if not np.any(juvenile_mask):
            # Some age-classified matrices contain reproductive contributions from all
            # source stages. In that case, retain a non-empty juvenile mask so that
            # juvenile-targeted perturbations remain well-defined.
            juvenile_mask = juvenile_mask.copy()
            juvenile_mask[0] = True

        object.__setattr__(self, "_adult_stage_mask", adult_mask.copy())
        object.__setattr__(self, "_juvenile_stage_mask", juvenile_mask.copy())

    @property
    def matrix(self) -> np.ndarray:
        """Return a copy of the projection matrix."""
        return self._A.copy()

    @property
    def n_stages(self) -> int:
        """Number of stages or classes in the projection matrix."""
        return self._A.shape[0]

    @property
    def fecundity_element_mask(self) -> np.ndarray:
        """Boolean matrix selecting fecundity entries."""
        return self._fecundity_mask.copy()

    @property
    def adult_stage_mask(self) -> np.ndarray:
        """Boolean vector selecting adult or reproductive source stages."""
        return self._adult_stage_mask.copy()

    @property
    def juvenile_stage_mask(self) -> np.ndarray:
        """Boolean vector selecting juvenile or pre-reproductive source stages."""
        return self._juvenile_stage_mask.copy()

    @property
    def lambda_(self) -> float:
        """Dominant eigenvalue of the unperturbed projection matrix."""
        return dominant_eigenvalue(self._A)

    def stable_distribution(self) -> np.ndarray:
        """Return the stable stage distribution of the model."""
        return stable_stage_distribution(self._A)
