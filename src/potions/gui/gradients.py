"""
Finite-difference gradients of the East River lateral-model
calibration GUI's objective functions.

The window's *Gradients* button differentiates every objective
function with respect to every free parameter by central
differences: each free parameter (a parameter whose current range
has a positive width) is nudged up and down by 1% of its current
range, clamped to the range's ends, and each perturbed point is run
through the same models as a normal run.  The objective vector of a
run is the run's objective values flattened: the streamflow block
first (when the run was scored against the measured streamflow),
then one block per measured river species, each block in the metric
order of `potions.gui.common.METRIC_NAMES`; a metric the run did
not report becomes NaN, so the difference of a missing value is
NaN.  The difference matrix has one row per objective function (the
target's name and the metric's display label, e.g. ``DOC: R²``) and
one column per free parameter; it is written to a
``gradients_YYYYMMDD_HHMMSS.csv`` file in the results directory and
drawn as an extra page of the figure panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

import potions as pt
from potions.result_saving import DEFAULT_RESULTS_DIR

from .common import METRIC_LABELS, METRIC_NAMES
from .data import GuiRunResults
from .state import objectives_from_results

#: The perturbation step as a fraction of a free parameter's current
#: range: the ± points of the central difference bracket the base
#: point by this amount of the range on each side.
STEP_FRACTION: Final[float] = 0.01


# ==== Perturbations ==== #


@dataclass(frozen=True, slots=True)
class Perturbation:
    """
    One free parameter's perturbation: its ± points and the
    effective step of each side.

    The points are fresh copies of the base parameter vector with
    this parameter's entry moved by `STEP_FRACTION` of the current
    range, clamped to the range's ends; the effective step is the
    actual distance from the base point, so a parameter sitting at a
    bound has a zero step on the out-of-range side and the
    difference there is one-sided.

    Attributes:
        key: The parameter's key.
        x_plus: The base vector with this parameter raised.
        x_minus: The base vector with this parameter lowered.
        h_plus: The effective step of the raised point (the
            distance from the base vector).
        h_minus: The effective step of the lowered point.
    """

    key: str
    x_plus: np.ndarray
    x_minus: np.ndarray
    h_plus: float
    h_minus: float


def free_parameter_steps(
    x: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> list[Perturbation]:
    """
    The ± perturbation of every free parameter around `x`.

    Free parameters are the ones whose current range has a positive
    width; a parameter with equal bounds cannot be perturbed and is
    omitted.  The perturbations come out in the bounds' key order,
    which is the order of the parameter vector.

    Args:
        x: The base parameter vector, in the bounds' key order.
        bounds: The current bounds of every parameter, in the
            parameter vector's key order.

    Returns:
        One perturbation per free parameter.
    """
    steps: list[Perturbation] = []
    for position, (key, (low, high)) in enumerate(bounds.items()):
        if high <= low:
            continue
        base: float = float(x[position])
        step: float = STEP_FRACTION * (high - low)
        up: float = min(base + step, high)
        down: float = max(base - step, low)
        x_plus: np.ndarray = np.array(x, dtype=np.float64, copy=True)
        x_plus[position] = up
        x_minus: np.ndarray = np.array(x, dtype=np.float64, copy=True)
        x_minus[position] = down
        steps.append(
            Perturbation(
                key=key,
                x_plus=x_plus,
                x_minus=x_minus,
                h_plus=up - base,
                h_minus=base - down,
            )
        )
    return steps


# ==== Objective vector ==== #


def objective_vector(
    results: pt.ModelResults | GuiRunResults,
) -> tuple[list[str], np.ndarray]:
    """
    The objective values of a run as a flat vector, with labels.

    The vector is `objectives_from_results` flattened: the
    streamflow block first (when the run was scored against the
    measured streamflow), then one block per measured river species
    in the reported order, each block in the metric order of
    `METRIC_NAMES`.  A metric the run did not report for a target
    becomes NaN, so the difference of a missing value is NaN.

    Args:
        results: The finished run's results (a `pt.ModelResults`
            for a full-mode run, a `GuiRunResults` for a
            hydrology-only run).

    Returns:
        The objective labels (the target's name and the metric's
        display label, e.g. ``DOC: R²``) and the vector of values,
        in the same order.
    """
    mapping: dict[str, dict[str, float]] = objectives_from_results(results)
    labels: list[str] = []
    values: list[float] = []
    for target, per_metric in mapping.items():
        for metric in METRIC_NAMES:
            labels.append(f"{target}: {METRIC_LABELS[metric]}")
            values.append(per_metric.get(metric, float("nan")))
    return labels, np.array(values, dtype=np.float64)


# ==== Difference matrix ==== #


def gradient_matrix(
    f_plus: list[np.ndarray],
    f_minus: list[np.ndarray],
    steps: list[Perturbation],
) -> np.ndarray:
    """
    The central-difference gradient matrix of a perturbation batch.

    Each entry is ``(f_plus - f_minus) / (h_plus + h_minus)`` for
    the step's parameter, with each step's ± objective vectors in
    the same order as `steps`; a NaN in either vector propagates
    into the entry.

    Args:
        f_plus: The objective vector of each step's raised point.
        f_minus: The objective vector of each step's lowered point.
        steps: The steps, in the same order as the vectors.

    Returns:
        The matrix with one row per objective function and one
        column per step, in the steps' order; an empty (0, 0)
        matrix when there are no steps.

    Raises:
        ValueError:
            Raised when the three sequences do not all have the
            same length.
    """
    if not (len(f_plus) == len(f_minus) == len(steps)):
        raise ValueError(
            "Expected the f_plus, f_minus, and steps sequences to "
            f"have equal lengths; got {len(f_plus)}, {len(f_minus)}, "
            f"and {len(steps)}."
        )
    if not steps:
        return np.empty((0, 0), dtype=np.float64)

    n_objectives: int = len(f_plus[0])
    matrix: np.ndarray = np.empty((n_objectives, len(steps)), dtype=np.float64)
    for column, (f_up, f_down, step) in enumerate(zip(f_plus, f_minus, steps)):
        matrix[:, column] = (f_up - f_down) / (step.h_plus + step.h_minus)
    return matrix


# ==== CSV export ==== #


def write_gradient_csv(
    table: pd.DataFrame,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """
    Write the gradient matrix to a timestamped CSV in the results
    directory.

    The file is named ``gradients_YYYYMMDD_HHMMSS.csv`` (local
    time, the same stamp convention as a run directory), so a
    write in the same second overwrites the earlier file.

    Args:
        table: The gradient matrix, one row per objective function
            and one column per parameter.
        results_dir: The results directory to write into; created
            when missing.  Defaults to the results directory of
            the current working directory.

    Returns:
        The path of the written file.
    """
    base_dir: Path = Path(results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path: Path = base_dir / f"gradients_{stamp}.csv"
    table.to_csv(path, index_label="objective")
    return path
