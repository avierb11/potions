"""
Helpers for the window's line search job.

The window's *Line search* button follows the steepest ascent of
one objective function from the current parameters: the direction
(the chosen objective's row of the last gradient job's difference
matrix, or the gradient computed by the same central differences
as the *Gradients* job) defines the straight ascent line
``x + t * direction``, and this module answers the three questions
that line raises: how a direction computed in the free
parameters' space becomes the parameter vector's space, which
values of ``t`` keep every parameter inside its current bounds,
and where the job's value profile gets written.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from potions.result_saving import DEFAULT_RESULTS_DIR

from .gradients import _Perturbation


def feasible_t_interval(
    x: np.ndarray,
    direction: np.ndarray,
    steps: list[_Perturbation],
    bounds: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    """
    The interval of ascent-line steps that stays inside the bounds.

    The ascent line is ``x + t * direction``; a step ``t`` is
    feasible when every free parameter's point
    ``x_i + t * direction_i`` lies inside its current range.  Each
    such parameter bounds ``t`` on one or both ends (a parameter
    whose direction component is zero does not move at all, and
    merely has to sit in bounds at the base point), and the
    feasible interval is the intersection of those bounds.  When
    the base point is in bounds, the interval contains ``t = 0``.

    Args:
        x: The base parameter vector, in the bounds' key order.
        direction: The unit-length ascent direction, in the
            parameter vector's order.
        steps: The free parameters' perturbations; their keys name
            the parameters that must stay in bounds.
        bounds: The current bounds of every parameter, in the
            parameter vector's key order.

    Returns:
        The (lower, upper) feasible interval of ``t``, or None
        when no step keeps every parameter inside its bounds.
    """
    positions: dict[str, int] = {key: pos for pos, key in enumerate(bounds)}
    t_lo: float = float("-inf")
    t_hi: float = float("inf")
    for step in steps:
        position: int = positions[step.key]
        (low, high) = bounds[step.key]
        base: float = float(x[position])
        delta: float = float(direction[position])
        if delta == 0.0:
            if not low <= base <= high:
                return None
            continue
        if delta > 0.0:
            t_lo = max(t_lo, (low - base) / delta)
            t_hi = min(t_hi, (high - base) / delta)
        else:
            t_lo = max(t_lo, (high - base) / delta)
            t_hi = min(t_hi, (low - base) / delta)
    if t_lo >= t_hi or not (np.isfinite(t_lo) and np.isfinite(t_hi)):
        return None
    return (t_lo, t_hi)


def expand_direction(
    row: np.ndarray,
    steps: list[_Perturbation],
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """
    The full parameter vector of a free-space direction row.

    A line search's direction is computed, or taken from a
    finished gradient job, as a row in the free parameters' space
    (one component per free parameter, in `steps`' order).  The
    search's arithmetic works in the full parameter vector's
    space, though, so this spreads the row's components into that
    space, at each free parameter's position in `bounds`, leaving
    the pinned parameters' components at zero (they never move).

    Args:
        row: The direction's components, in `steps`' order.
        steps: The free parameters' perturbations; their keys name
            the parameters the row's components belong to, in the
            same order as the row's components.
        bounds: The parameter vector's bounds, in its key order.

    Returns:
        The direction as a full-length parameter vector.
    """
    positions: dict[str, int] = {key: pos for pos, key in enumerate(bounds)}
    full: np.ndarray = np.zeros(len(bounds), dtype=np.float64)
    for step, component in zip(steps, row):
        full[positions[step.key]] = component
    return full


def write_line_search_csv(
    profile: pd.DataFrame,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """
    Write the line search's value profile to a timestamped CSV in
    the results directory.

    The file is named ``line_search_YYYYMMDD_HHMMSS.csv`` (local
    time, the same stamp convention as a run directory), so a
    write in the same second overwrites the earlier file.

    Args:
        profile: The value profile, one row per evaluated point:
            the step ``t`` as index, the objective's value and
            every free parameter's value as columns.
        results_dir: The results directory to write into; created
            when missing.  Defaults to the results directory of
            the current working directory.

    Returns:
        The path of the written file.
    """
    base_dir: Path = Path(results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path: Path = base_dir / f"line_search_{stamp}.csv"
    profile.to_csv(path, index_label="t")
    return path
