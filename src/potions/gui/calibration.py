"""
Hydrological differential-evolution calibration for the GUI.

The window's *Calibrate* button optimizes the free hydrology
parameters against the objective terms checked in the objective
table.  Only hydrology parameters whose current cell range has a
positive width are searched; a pinned parameter keeps its current
value.  The objective is the equal-weighted sum of the selected
terms, maximized (the optimizer minimizes its negative), exactly like
the line search's objective.

The job runs in a worker thread through
`HydrologicalModel.simple_calibration`; the optimizer's generation
callback feeds the window's progress bar.  The best run is returned to
the window, which adopts its parameters into the panel, refreshes the
figures, and saves the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Final, Optional

import numpy as np
import potions as pt
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from potions.gui.data import GuiInputData, GuiRunResults
from potions.hydro_model import HydrologicalModel

# ==== Constants ==== #
# The objective-table metric labels the calibration can optimize, and
# the keys of the results' objective-function series they name.
_METRIC_KEYS: Final[dict[str, str]] = {
    "KGE": "kge",
    "LogKGE": "log_kge",
    "NSE": "nse",
    "LogNSE": "log_nse",
}

# The defaults offered by the calibration dialog.
_DEFAULT_GENERATIONS: Final[int] = 50
_DEFAULT_POPULATION_SIZE: Final[int] = 15
# =================== #


# ==== Types ==== #
@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    """
    The user-settable differential-evolution options.

    Attributes:
        generations: The maximum number of generations
            (``maxiter`` in SciPy's terminology).
        population_size: The population-size multiplier
            (``popsize``).
        polish: Whether to refine the best point with a local
            optimizer.
    """

    generations: int = _DEFAULT_GENERATIONS
    population_size: int = _DEFAULT_POPULATION_SIZE
    polish: bool = False


@dataclass(frozen=True, slots=True)
class _CalibrationOutcome:
    """
    The outcome of one finished calibration job.

    A successful job carries the best full parameter vector, the
    best run's results, the objective terms it optimized, and the
    best objective value (the sum of the selected terms, higher is
    better); a failed job carries the error instead, with the vector
    and results None.

    Attributes:
        error: The failure, formatted as ``Name: message``; None on
            success.
        x_best: The full parameter vector of the best point, in the
            window's parameter order; None on failure.
        results: The best run's results; None on failure.
        terms: The objective terms the job optimized, as
            ``"target: metric"`` labels.
        best_value: The best objective value (the equal-weighted
            sum of the selected terms); NaN on failure.
        evaluations: The number of objective evaluations the
            optimizer reported; zero on failure.
        elapsed_seconds: The wall-clock duration of the job.
        model_key: The key of the model spec the job ran.
    """

    error: str | None
    x_best: NDArray[np.float64] | None
    results: GuiRunResults | None
    terms: list[str]
    best_value: float
    evaluations: int
    elapsed_seconds: float
    model_key: str


class _CalibrationWorker(QThread):
    """
    A worker thread running one differential-evolution calibration.

    The worker expands the free hydrology parameters with the
    current values of the pinned ones, optimizes the selected
    objective terms, and returns the best full parameter vector and
    run to the window.  Its generation callback reports the
    optimizer's progress on the window's status bar and progress
    bar; on completion it emits ``finished_calibration``.

    Signals:
        finished_calibration: Emitted (on the main thread) with the
            `_CalibrationOutcome`.
        progress_line: Emitted with each generation's progress line.
        progress_step: Emitted with (done, total) generation counts.

    Args:
        x: The current full parameter vector, in the bounds' key
            order.
        input_data: The assembled calibration inputs.
        bounds: The current bounds of every parameter, in the
            parameter vector's order.
        terms: The objective terms to optimize, as
            ``"target: metric"`` labels.
        settings: The optimizer options.
        model_key: The key of the model spec the window shows; the
            window discards an outcome from another model.
        parent: The parent object (the window); defaults to None.
    """

    finished_calibration = Signal(object)
    progress_line = Signal(str)
    progress_step = Signal(int, int)

    def __init__(
        self,
        x: NDArray[np.float64],
        input_data: GuiInputData,
        bounds: dict[str, tuple[float, float]],
        terms: list[str],
        settings: CalibrationSettings,
        model_key: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._x: NDArray[np.float64] = np.array(x, dtype=np.float64)
        self._input_data: GuiInputData = input_data
        self._bounds: dict[str, tuple[float, float]] = bounds
        self._terms: list[str] = list(terms)
        self._settings: CalibrationSettings = settings
        self._model_key: str = model_key

    def run(self) -> None:
        """Run the calibration and emit its outcome."""
        self.finished_calibration.emit(self._compute())

    def _compute(self) -> _CalibrationOutcome:
        """
        Run the optimizer and build the job's outcome.

        Returns:
            The successful outcome, or the failed one carrying the
            exception's formatted message.
        """
        started: float = time.perf_counter()
        try:
            model_class: type[HydrologicalModel] = (
                self._input_data.spec.model_class
            )
            num_hydro: int = model_class.get_num_hydro_parameters()
            all_names: tuple[str, ...] = tuple(self._input_data.bounds)[
                :num_hydro
            ]
            hydro_values: dict[str, float] = {
                name: float(self._x[index])
                for index, name in enumerate(all_names)
            }
            free_ranges: dict[str, tuple[float, float]] = {
                name: (
                    float(self._bounds[name][0]),
                    float(self._bounds[name][1]),
                )
                for name in all_names
                if self._bounds[name][1] > self._bounds[name][0]
            }
            if not free_ranges:
                raise ValueError(
                    "No free hydrology parameters to calibrate."
                )
            metric_keys: tuple[str, ...] = metric_keys_for_terms(
                self._terms
            )
            if not metric_keys:
                raise ValueError(
                    "No calibratable objective terms are selected."
                )

            generation: int = 0
            total: int = self._settings.generations

            def callback(xk: NDArray[np.float64], convergence: float) -> None:
                nonlocal generation
                generation += 1
                self.progress_step.emit(generation, total)
                self.progress_line.emit(
                    f"generation {generation}/{total} "
                    f"(convergence {convergence:.3g})"
                )

            self.progress_step.emit(0, total)
            opt_params, best_results, opt_res = (
                model_class.simple_calibration(
                    forc=self._input_data.forc,
                    meas_streamflow=self._input_data.q_meas,
                    metric=partial(
                        objective_score, metric_keys=metric_keys
                    ),
                    num_threads=-1,
                    polish=self._settings.polish,
                    maxiter=self._settings.generations,
                    param_ranges=free_ranges,
                    popsize=self._settings.population_size,
                    base_params=hydro_values,
                    callback=callback,
                )
            )

            x_best: NDArray[np.float64] = self._x.copy()
            for index, name in enumerate(all_names):
                x_best[index] = float(
                    opt_params.get(name, hydro_values[name])
                )
            return _CalibrationOutcome(
                error=None,
                x_best=x_best,
                results=GuiRunResults(hydro=best_results, rt=None),
                terms=list(self._terms),
                best_value=-float(opt_res.fun),
                evaluations=int(getattr(opt_res, "nfev", 0)),
                elapsed_seconds=time.perf_counter() - started,
                model_key=self._model_key,
            )
        except Exception as err:
            return _CalibrationOutcome(
                error=f"{type(err).__name__}: {err}",
                x_best=None,
                results=None,
                terms=list(self._terms),
                best_value=float("nan"),
                evaluations=0,
                elapsed_seconds=time.perf_counter() - started,
                model_key=self._model_key,
            )


class CalibrationDialog(QDialog):
    """
    The small dialog that asks for the calibration's optimizer options.

    The dialog carries a spin box for the generation count and one
    for the population-size multiplier, plus a *Polish* checkbox.
    The defaults are `CalibrationSettings`'s.

    Args:
        parent: The dialog's parent widget; defaults to None.
        settings: The values the controls start at; defaults to the
            standard `CalibrationSettings`.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        settings: Optional[CalibrationSettings] = None,
    ) -> None:
        super().__init__(parent)
        initial: CalibrationSettings = settings or CalibrationSettings()
        self.setWindowTitle("Calibrate hydrology")

        self._generations: QSpinBox = QSpinBox()
        self._generations.setRange(1, 100_000)
        self._generations.setValue(initial.generations)

        self._population_size: QSpinBox = QSpinBox()
        self._population_size.setRange(5, 10_000)
        self._population_size.setValue(initial.population_size)

        self._polish: QCheckBox = QCheckBox(
            "Polish the best point with a local optimizer"
        )
        self._polish.setChecked(initial.polish)

        form: QFormLayout = QFormLayout()
        form.addRow("Generations", self._generations)
        form.addRow("Population size", self._population_size)
        form.addRow(self._polish)

        buttons: QDialogButtonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def settings(self) -> CalibrationSettings:
        """
        The options currently shown by the controls.

        Returns:
            The `CalibrationSettings` the user entered.
        """
        return CalibrationSettings(
            generations=self._generations.value(),
            population_size=self._population_size.value(),
            polish=self._polish.isChecked(),
        )


# ==== Functions ==== #
def metric_keys_for_terms(terms: list[str]) -> tuple[str, ...]:
    """
    Map objective-table terms onto the results' metric keys.

    Args:
        terms: The terms to map, as ``"target: metric"`` labels.

    Returns:
        The unique metric keys of the terms the calibration supports,
        in the terms' order.
    """
    keys: list[str] = []
    for term in terms:
        parts: list[str] = term.split(": ", 1)
        if len(parts) != 2:
            continue
        key: Optional[str] = _METRIC_KEYS.get(parts[1])
        if key is not None and key not in keys:
            keys.append(key)
    return tuple(keys)


def objective_score(
    results: pt.HydroModelResults,
    metric_keys: tuple[str, ...],
) -> float:
    """
    The negative equal-weighted sum of a run's selected metrics.

    Differential evolution minimizes its objective, so the sum the
    window maximizes is negated here.  A missing or non-finite metric
    makes the point unscoreable and returns infinity.

    Args:
        results: The run's hydrology results.
        metric_keys: The objective-function keys to sum.

    Returns:
        The negated sum of the selected metrics, or infinity when a
        metric is missing or not finite.
    """
    try:
        values: list[float] = [
            float(results.objective_functions[key]) for key in metric_keys
        ]
    except (KeyError, TypeError, ValueError):
        return float("inf")
    if not all(np.isfinite(values)):
        return float("inf")
    return -float(sum(values))


def ask_calibration_settings(
    parent: QWidget | None = None,
) -> Optional[CalibrationSettings]:
    """
    Ask the user for the calibration's optimizer options.

    Args:
        parent: The dialog's parent widget; defaults to None.

    Returns:
        The entered settings, or None when the dialog was cancelled.
    """
    dialog: CalibrationDialog = CalibrationDialog(parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.settings()
# =============== #
