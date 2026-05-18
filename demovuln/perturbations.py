from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .models import MatrixPopulationModel

Target = Literal["adult_survival", "juvenile_survival", "fecundity", "all", "custom"]


def _column_mask_to_element_mask(column_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:, column_mask] = True
    return mask


def build_target_mask(
    model: MatrixPopulationModel,
    target: Target,
    *,
    survival_affects_fecundity: bool = True,
    custom_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct the matrix-entry mask associated with a perturbation target.

    Parameters
    ----------
    model : MatrixPopulationModel
        Matrix population model defining the projection matrix and the mapping
        between matrix entries and demographic components.
    target : {"adult_survival", "juvenile_survival", "fecundity", "all", "custom"}
        Demographic component to perturb. Survival targets are defined by
        source-stage columns; fecundity targets are defined by
        ``model.fecundity_element_mask``.
    survival_affects_fecundity : bool, default=True
        If True, survival perturbations applied to adult or juvenile source
        stages scale the full source-stage column, including fecundity entries.
        If False, fecundity entries are excluded from survival targets.
    custom_mask : array_like of bool, optional
        Boolean matrix selecting entries to perturb when ``target="custom"``.
        It must have the same shape as the projection matrix.

    Returns
    -------
    numpy.ndarray of bool
        Boolean matrix with ``True`` at entries affected by the perturbation.

    Raises
    ------
    ValueError
        If ``target="custom"`` is used without a valid ``custom_mask``, or if
        ``target`` is not recognized.
    """
    A = model.matrix
    fec_mask = model.fecundity_element_mask

    if target == "custom":
        if custom_mask is None:
            raise ValueError("custom_mask must be supplied when target='custom'.")
        mask = np.asarray(custom_mask, dtype=bool)
        if mask.shape != A.shape:
            raise ValueError("custom_mask must have the same shape as the model matrix.")
        return mask

    if target == "fecundity":
        return fec_mask

    if target == "adult_survival":
        mask = _column_mask_to_element_mask(model.adult_stage_mask, A.shape)
    elif target == "juvenile_survival":
        mask = _column_mask_to_element_mask(model.juvenile_stage_mask, A.shape)
    elif target == "all":
        mask = A > 0
    else:
        raise ValueError(f"Unknown perturbation target: {target!r}.")

    if not survival_affects_fecundity and target in {"adult_survival", "juvenile_survival"}:
        mask = mask & ~fec_mask
    return mask


def apply_perturbation(
    model: MatrixPopulationModel,
    target: Target,
    magnitude: float,
    *,
    survival_affects_fecundity: bool = True,
    custom_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a multiplicative demographic perturbation to a projection matrix.

    Parameters
    ----------
    model : MatrixPopulationModel
        Matrix population model defining the unperturbed projection matrix and
        demographic target definitions.
    target : {"adult_survival", "juvenile_survival", "fecundity", "all", "custom"}
        Demographic component affected by the perturbation.
    magnitude : float
        Proportional reduction applied to the selected entries. A value of
        ``0`` leaves the matrix unchanged, whereas ``1`` sets the selected
        entries to zero.
    survival_affects_fecundity : bool, default=True
        If True, survival perturbations scale whole source-stage columns,
        including fecundity entries. This represents perturbations acting on the
        contribution of individuals in a source stage. If False, fecundity
        entries are not modified by survival targets.
    custom_mask : array_like of bool, optional
        Boolean matrix selecting entries to perturb when ``target="custom"``.

    Returns
    -------
    numpy.ndarray
        Perturbed projection matrix.

    Notes
    -----
    The perturbation is implemented as ``a_ij -> (1 - m) a_ij`` for selected
    entries, where ``m`` is ``magnitude``. For ``target="all"`` with
    ``survival_affects_fecundity=True``, fecundity entries receive two
    multiplicative reductions, ``(1 - m)^2``, representing simultaneous effects
    on reproductive output and on the survival or availability of reproductive
    source stages.
    """
    if magnitude < 0 or magnitude > 1:
        raise ValueError("magnitude must lie in [0, 1].")

    B = model.matrix
    factor = 1.0 - float(magnitude)

    if target == "all" and survival_affects_fecundity:
        fec_mask = model.fecundity_element_mask
        nonfec_mask = (B > 0) & ~fec_mask
        B[fec_mask] *= factor**2
        B[nonfec_mask] *= factor
        return B

    mask = build_target_mask(
        model,
        target,
        survival_affects_fecundity=survival_affects_fecundity,
        custom_mask=custom_mask,
    )
    B[mask] *= factor
    return B
