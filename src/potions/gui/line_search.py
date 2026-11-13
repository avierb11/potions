"""
Helpers for the window's line search job.

The window's *Line search* button follows the steepest ascent of
one objective function from the current parameters: the direction
(the chosen objective's row of the last gradient job's difference
matrix, or the gradient computed by the same forward differences
as the *Gradients* job) defines the straight ascent line
``x + t * direction``, and this module adapts that line's geometry
to the shared wave-search engine in `potions.line_search`.  The
window describes its parameters as `Perturbation` steps and a
bounds mapping; the engine works in raw vector positions and bound
pairs, so each helper here translates the window's description and
delegates the arithmetic: turning a free-space direction row into
the parameter vector's space, projecting a direction so its steps
never leave the current bounds, and finding the steps that keep
every parameter inside those bounds.  Where the value profile gets
written is this module's own concern.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from potions.result_saving import DEFAULT_RESULTS_DIR

from .. import line_search as engine
from .gradients import Perturbation

search_t_points = engine.search_t_points


def feasible_t_interval(
    x: np.ndarray,
    direction: np.ndarray,
    steps: list[Perturbation],
    bounds: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    """
    The interval of ascent-line steps that stays inside the bounds.

    Translates the window's steps and bounds into the engine's
    positions and bound pairs and defers to
    `engine.feasible_t_interval`: each free parameter bounds the
    step ``t`` of the line ``x + t * direction`` on one or both
    ends, and the feasible interval is the intersection of those
    bounds.

    Args:
        x: The base parameter vector, in the bounds' key order.
        direction: The unit-length ascent direction, in the
            parameter vector's order.
        steps: The free parameters' perturbations; their keys name
            the parameters that must stay in bounds.
        bounds: The current bounds of every parameter, in the
            parameter vector's key order.

    Returns:
        The (lower, upper) feasible interval of ``t``, or None when
        no step keeps every parameter inside its bounds.
    """
    return engine.feasible_t_interval(
        x,
        direction,
        [list(bounds).index(step.key) for step in steps],
        [bounds[step.key] for step in steps],
    )


def project_direction(
    x: np.ndarray,
    direction: np.ndarray,
    steps: list[Perturbation],
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """
    The ascent direction with every out-of-bounds component dropped.

    Defers to `engine.project_direction` (see it for the projection
    rules and the bound-sitting tolerance), passing the free
    parameters' keys as names so a failure names the blocked
    parameters rather than their vector positions.

    Args:
        x: The base parameter vector, in the bounds' key order.
        direction: The ascent direction, in the parameter
            vector's order.
        steps: The free parameters' perturbations; their keys name
            the parameters whose components may need dropping.
        bounds: The current bounds of every parameter, in the
            parameter vector's key order.

    Returns:
        The projected direction as a unit-length parameter vector.

    Raises:
        ValueError:
            Raised when no component survives the projection: every
            free parameter that can move sits on the bound its
            component points out of, or the direction carries no
            component along any free parameter.
    """
    return engine.project_direction(
        x,
        direction,
        [list(bounds).index(step.key) for step in steps],
        [bounds[step.key] for step in steps],
        names=[step.key for step in steps],
    )


def expand_direction(
    row: np.ndarray,
    steps: list[Perturbation],
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    """
    The full parameter vector of a free-space direction row.

    Translates the window's steps into the engine's positions and
    defers to `engine.expand_direction`: the row's components land
    at each free parameter's position in the parameter vector, and
    the pinned parameters' components stay at zero (they never
    move).

    Args:
        row: The direction's components, in `steps`' order.
        steps: The free parameters' perturbations; their keys name
            the parameters the row's components belong to, in the
            same order as the row's components.
        bounds: The parameter vector's bounds, in its key order.

    Returns:
        The direction as a full-length parameter vector.
    """
    return engine.expand_direction(
        row,
        [list(bounds).index(step.key) for step in steps],
        len(bounds),
    )


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
