"""
The main window of the East River lateral-model calibration GUI.

The window stacks two splitter sections: the figure panel (the
model's figures, one visible at a time, flipped by the panel's
arrow buttons or the main window's Left / Right keys: the three
full-mode figures, or the single streamflow figure in
hydrology-only mode, plus the gradient job's matrix as an extra
page when one has finished) and, below it, an inner splitter
holding the per-run objective-function table on the left and the
parameter panel on the right.

Parameter changes re-run the models on a worker thread (a short
debounce covers a drag or a keyboard step) and refresh the figures
and the table when the run finishes; a generation counter discards
stale results if the parameters change mid-run.  Plot-selection
changes rebuild the figures from the last results without
re-running.  The header's *Auto-run* checkbox (checked by default)
switches the edit-triggered re-runs off, so a batch of edits can
wait for one manual run from the *Run model* button.  The header's
*Hydrology only* checkbox (off by default) switches the run to
hydrology-only mode: only the hydrology model is run, scored
against the measured streamflow, the reactive-transport parameter
groups are hidden, the figures reduce to the single streamflow
figure, and the objective table carries only the streamflow column.

The header's *Gradients* button differentiates every objective
function with respect to every free parameter (a parameter whose
range has a positive width) by central differences: each
parameter is nudged up and down by 1% of its current range,
clamped to the range's ends, and every perturbed point — plus the
base point when the results on screen are not up to date — is run
through the models on a worker thread.  The resulting matrix, one
row per objective function and one column per free parameter, is
written to a ``gradients_YYYYMMDD_HHMMSS.csv`` file in the results
directory and drawn as an extra page of the figure panel.

The parameter panel's *History* group lists the previous runs —
the saved run directories and this session's runs, newest first —
with an *Overlay previous results* checkbox (the selected run's
results are drawn faintly on the current figures) and a *Use
these parameters* button that restores a run's parameters, mode,
and results to the screen without a re-run.

At startup the window restores the most recent saved run: parameter
values,
edited bounds, plot selections, the window geometry, and the
splitter sizes come from the run's ``gui_state.json`` (or, for an
older run, best-effort from its parameters CSVs), and the figures
are built straight from the run's saved simulation output, so the
window opens without a re-run.  With no saved run it starts from
the default parameters and runs the models once in the background.

A *Save run* click writes the existing per-run model output plus a
``gui_state.json`` capturing the session state, so the next start
restores exactly what the user saw.  With the header's *Save run
history* checkbox on, every finished run also appends one
abbreviated record (timestamp, mode, parameter values, objective
values) to a ``gui_run_history.jsonl`` file in the results
directory.
"""

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from io import TextIOBase
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import potions as pt
from matplotlib.figure import Figure
from potions.hydro_model import HydrologicalModel
from potions.result_saving import save_hydro_run, save_model_run
from PySide6.QtCore import (
    QAbstractTableModel,
    QCoreApplication,
    QObject,
    QModelIndex,
    QPersistentModelIndex,
    QRect,
    QThread,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QCloseEvent, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableView,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from potions.gui.common import METRIC_NOTES
from potions.gui.data import (
    GuiInputData,
    GuiRunResults,
    PlotSelection,
    arr_to_hydro_results,
    arr_to_model,
    arr_to_results,
    default_plot_selection,
)
from potions.gui.figure_panel import FigurePanel
from potions.gui.gradients import (
    _Perturbation,
    free_parameter_steps,
    gradient_matrix,
    objective_vector,
    write_gradient_csv,
)
from potions.gui.param_panel import ParamPanel
from potions.gui.plots import (
    build_objective_table,
    make_gradient_figure,
    make_rate_figure,
    make_time_series_figure,
    make_zone_figure,
)
from potions.gui.state import (
    MODE_FULL,
    MODE_HYDRO,
    GuiState,
    RunHistoryEntry,
    SavedRun,
    append_run_history,
    list_run_dirs,
    load_saved_run,
    objectives_from_results,
    write_gui_state,
)

# ==== Constants ==== #
# The window title (the title row repeats it in a larger font).
_TITLE: Final[str] = "East River lateral model calibration"

# The debounce interval (ms) between a parameter change and the
# re-run; a drag or a burst of keyboard steps coalesces into one.
_DEBOUNCE_MS: Final[int] = 400

# The interval (ms) at which the elapsed-time label refreshes while
# a run is in flight.
_ELAPSED_MS: Final[int] = 200

# The fixed width (px) of the indeterminate progress bar.
_PROGRESS_WIDTH: Final[int] = 120

# The initial window size (px); a restored window geometry replaces
# it before the window is shown.
_INITIAL_SIZE: Final[tuple[int, int]] = (1200, 900)

# The initial sizes (px) of the window's two splitters, in order:
# the figure and the lower (table-and-parameters) row of the main
# vertical splitter, then the objective-table and parameter-panel
# widths of the inner horizontal splitter.  Qt scales each pair
# proportionally to the splitter's size on the first layout, and a
# restored session state replaces them with its own.
_INITIAL_SPLIT_SIZES: Final[tuple[int, int, int, int]] = (350, 530, 300, 880)

# The figure panel's minimum height (px): even a restored state or a
# manual drag can then never squash the plot into a sliver.
_FIGURE_MIN_HEIGHT: Final[int] = 240

# The most recent in-session runs the History dropdown keeps (the
# session's own runs, newest first; the saved run directories listed
# below them are not capped).
_HISTORY_CAP: Final[int] = 20

# The placeholder for an objective cell whose value is missing or
# NaN.
_MISSING: Final[str] = "–"

# An invalid QModelIndex, for the model methods' default parents.
_INVALID_INDEX: Final[QModelIndex] = QModelIndex()
# =================== #


# ==== Helpers ==== #
class _CapturingStream:
    """
    A stdout stand-in that mirrors writes to the real stdout and
    forwards each completed line to a callback.

    The worker installs it around a model run so the models'
    verbose progress lines (``Starting hydrologic simulation…`` and
    friends) reach the window's status bar without blocking; the
    real stdout stays in sync, so console output is unchanged.

    Args:
        on_line: Called with each completed line (no trailing
            newline).
        fallback: The real stdout; every write is mirrored to it.
    """

    def __init__(
        self, on_line: Callable[[str], None], fallback: TextIOBase
    ) -> None:
        self._on_line: Callable[[str], None] = on_line
        self._fallback: TextIOBase = fallback
        self._buffer: str = ""

    def write(self, text: str) -> None:
        """Mirror `text` to the real stdout and emit each complete line."""
        self._fallback.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line: str
            line, self._buffer = self._buffer.split("\n", 1)
            self._on_line(line)

    def flush(self) -> None:
        """Flush the real stdout and emit any trailing partial line."""
        self._fallback.flush()
        if self._buffer:
            self._on_line(self._buffer)
            self._buffer = ""


@dataclass(frozen=True, slots=True)
class _RunOutcome:
    """
    The outcome of one worker run, delivered on the main thread
    through the worker's ``finished_run`` signal.

    Attributes:
        generation: The run's generation (the worker's identity);
            the window discards an outcome whose generation no
            longer matches the latest started run.
        results: The run's `GuiRunResults`, or None when the run
            failed.
        error: A one-line description of the failure, or None.
        elapsed_seconds: The wall-clock duration of the run.
    """

    generation: int
    results: GuiRunResults | None
    error: str | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _HistoryItem:
    """
    One previous run in the window's history, behind the parameter
    panel's *History* dropdown.

    A session run (one finished in this session) carries its
    parameters, its results, and its mode in memory; a saved run
    directory carries only its path and its label and is re-read
    from the run's saved state on demand.

    Attributes:
        label: The text shown in the dropdown: the run's name
            (``Run 3 — 14:22:31 (full)`` for a session run, the
            directory's name for a saved run).
        run_dir: The saved run directory, for a saved run; None for
            a session run.
        param_values: The run's parameter values, by key (the
            session's own values; None for a saved run, whose values
            come from its saved state on demand).
        results: The run's `GuiRunResults` (the overlay's frames;
            None for a saved run, which is re-read on demand).
        mode: The run's mode (`MODE_FULL` or `MODE_HYDRO`); None for
            a saved run, whose mode comes from its saved state.
    """

    label: str
    run_dir: Path | None
    param_values: dict[str, float] | None
    results: GuiRunResults | None
    mode: str | None


# ==== Worker ==== #
class _RunWorker(QThread):
    """
    A worker thread running the models off the event loop: the
    hydrology and reactive-transport models together in full mode,
    or the hydrology model alone in hydrology-only mode.

    The worker mirrors its stdout to the real one (so console
    output is unchanged) and forwards each completed line to the
    window's status bar; on completion it emits ``finished_run``
    with a `_RunOutcome`.  Runs cannot be interrupted, so the
    window waits for a worker before closing.

    Signals:
        finished_run: Emitted (on the main thread) with the
            `_RunOutcome` of this run.
        progress_line: Emitted with each completed stdout line of
            the run, for the window's status bar.

    Args:
        generation: The run's generation counter.
        x: The parameter vector to run with.
        input_data: The assembled calibration inputs.
        mode: The run mode (`MODE_FULL` or `MODE_HYDRO`): which
            models the run executes.
        parent: The parent object (the window); defaults to None.
    """

    finished_run = Signal(object)
    progress_line = Signal(str)

    def __init__(
        self,
        generation: int,
        x: np.ndarray,
        input_data: GuiInputData,
        mode: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._generation: int = generation
        self._x: np.ndarray = x
        self._input_data: GuiInputData = input_data
        self._mode: str = mode

    def run(self) -> None:
        """Run the models and emit the outcome, capturing stdout lines."""
        # The Any annotation is the type of sys.stdout in the
        # stdlib stubs; the value is a real text stream at runtime.
        real_stdout: Any = sys.stdout
        sys.stdout = _CapturingStream(self.progress_line.emit, real_stdout)
        started: float = time.perf_counter()
        results: GuiRunResults | None = None
        error: str | None = None
        try:
            if self._mode == MODE_HYDRO:
                results = arr_to_hydro_results(self._x, self._input_data)
            else:
                results = arr_to_results(self._x, self._input_data)
        except Exception as err:
            error = f"{type(err).__name__}: {err}"
        finally:
            sys.stdout = real_stdout
        self.finished_run.emit(
            _RunOutcome(
                self._generation, results, error, time.perf_counter() - started
            )
        )


# ==== Gradient worker ==== #


@dataclass(frozen=True, slots=True)
class _GradientOutcome:
    """
    The outcome of one finished gradient job.

    A successful job carries the difference matrix (one row per
    objective function, one column per free parameter), the base
    point's results (so the window can adopt them when they match
    its current parameters and mode), and the path of the CSV the
    matrix was written to; a failed job carries the error instead,
    with the matrix and the CSV path None (and the results None as
    well when the base point's own run failed).

    Attributes:
        error: The failure, formatted as ``Name: message``; None
            on success.
        table: The difference matrix with the objective labels as
            index and the parameter keys as columns; None on
            failure.
        base_results: The base point's results; None when the base
            point's own run failed.
        base_x: The base parameter vector the job differentiated
            around.
        mode: The run mode the job's points ran in (`MODE_FULL` or
            `MODE_HYDRO`).
        csv_path: The path of the written matrix CSV; None on
            failure.
        elapsed_seconds: The wall-clock duration of the job.
    """

    error: str | None
    table: pd.DataFrame | None
    base_results: GuiRunResults | None
    base_x: np.ndarray
    mode: str
    csv_path: Path | None
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _GradientSnapshot:
    """
    The last finished gradient job's results, kept for the figure
    panel.

    The matrix is re-drawn as the figure panel's extra page every
    time the figures are rebuilt, so it stays on screen across
    runs, mode changes, and plot-selection changes until the next
    successful gradient job replaces it.

    Attributes:
        table: The difference matrix (the objective labels as
            index, the parameter keys as columns).
        mode: The run mode the job's points ran in (`MODE_FULL` or
            `MODE_HYDRO`), shown in the page's title.
        csv_path: The path of the written matrix CSV, quoted in
            the page's caption.
        elapsed_seconds: The wall-clock duration of the job, shown
            in the completion status line.
    """

    table: pd.DataFrame
    mode: str
    csv_path: Path
    elapsed_seconds: float


class _GradientWorker(QThread):
    """
    A worker thread computing the finite-difference gradient of
    every objective function with respect to every free parameter.

    Each free parameter is perturbed up and down by 1% of its
    current range (clamped to the range's ends) and each
    perturbed point is run through the models in the job's run
    mode; the base point reuses the window's results when they
    were up to date at the job's start and is run otherwise.  The
    worker mirrors its stdout to the real one (so console output
    is unchanged) and forwards its progress lines to the window's
    status bar; on completion it emits ``finished_gradient`` with
    a `_GradientOutcome`.  Jobs cannot be interrupted, so the
    window waits for a worker before closing.

    Signals:
        finished_gradient: Emitted (on the main thread) with the
            `_GradientOutcome` of this job.
        progress_line: Emitted with the job's progress lines (the
            base point, then each perturbed point), for the
            window's status bar.

    Args:
        base_x: The parameter vector to differentiate around.
        steps: The ± perturbation of every free parameter, in the
            parameter vector's key order.
        mode: The run mode (`MODE_FULL` or `MODE_HYDRO`) the
            points run in.
        input_data: The assembled calibration inputs.
        base_results: The base point's results when they are
            already up to date on screen (None to run the base
            point too).
        parent: The parent object (the window); defaults to None.
    """

    finished_gradient = Signal(object)
    progress_line = Signal(str)

    def __init__(
        self,
        base_x: np.ndarray,
        steps: list[_Perturbation],
        mode: str,
        input_data: GuiInputData,
        base_results: GuiRunResults | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._base_x: np.ndarray = base_x
        self._steps: list[_Perturbation] = steps
        self._mode: str = mode
        self._input_data: GuiInputData = input_data
        self._base_results: GuiRunResults | None = base_results

    @staticmethod
    def _key(point: np.ndarray) -> tuple[float, ...]:
        """The identity of a parameter vector, for the results cache."""
        return tuple(float(value) for value in point)

    def _run_point(self, point: np.ndarray) -> GuiRunResults:
        """Run the models for `point` in this job's run mode."""
        if self._mode == MODE_HYDRO:
            return arr_to_hydro_results(point, self._input_data)
        return arr_to_results(point, self._input_data)

    def run(self) -> None:
        """Run the points of the job and emit the outcome,
        capturing stdout lines."""
        # The Any annotation is the type of sys.stdout in the
        # stdlib stubs; the value is a real text stream at runtime.
        real_stdout: Any = sys.stdout
        sys.stdout = _CapturingStream(self.progress_line.emit, real_stdout)
        started: float = time.perf_counter()
        outcome: _GradientOutcome
        try:
            outcome = self._compute(started)
        except Exception as err:
            outcome = _GradientOutcome(
                error=f"{type(err).__name__}: {err}",
                table=None,
                base_results=self._base_results,
                base_x=self._base_x,
                mode=self._mode,
                csv_path=None,
                elapsed_seconds=time.perf_counter() - started,
            )
        finally:
            sys.stdout = real_stdout
        self.finished_gradient.emit(outcome)

    def _compute(self, started: float) -> _GradientOutcome:
        """
        Run the job's points and build its success outcome.

        The base point is run only when it was not handed in; the
        perturbed points are deduplicated against it (a point
        clamped at a range end coincides with the base) and each
        distinct point is run once, in the steps' order.

        Args:
            started: The wall-clock time the job started, for the
                outcome's elapsed time.

        Returns:
            The success outcome: the difference matrix, the base
            point's results, and the path of the written CSV.
        """
        total: int = len(self._steps) * 2
        if self._base_results is not None:
            base_results: GuiRunResults = self._base_results
        else:
            self.progress_line.emit(f"Gradient run 0/{total}: base")
            base_results = self._run_point(self._base_x)
        results: dict[tuple[float, ...], GuiRunResults] = {}
        results[self._key(self._base_x)] = base_results

        seen: set[tuple[float, ...]] = {self._key(self._base_x)}
        pending: list[tuple[str, np.ndarray]] = []
        for step in self._steps:
            for suffix, point in ((" +", step.x_plus), (" -", step.x_minus)):
                key: tuple[float, ...] = self._key(point)
                if key not in seen:
                    seen.add(key)
                    pending.append((f"{step.key}{suffix}", point))

        for index, (label, point) in enumerate(pending, start=1):
            self.progress_line.emit(f"Gradient run {index}/{total}: {label}")
            results[self._key(point)] = self._run_point(point)

        base_labels, _ = objective_vector(base_results)
        f_plus: list[np.ndarray] = []
        f_minus: list[np.ndarray] = []
        for step in self._steps:
            _, up_vector = objective_vector(results[self._key(step.x_plus)])
            _, down_vector = objective_vector(results[self._key(step.x_minus)])
            f_plus.append(up_vector)
            f_minus.append(down_vector)

        matrix: np.ndarray = gradient_matrix(f_plus, f_minus, self._steps)
        table: pd.DataFrame = pd.DataFrame(
            matrix,
            index=base_labels,
            columns=[step.key for step in self._steps],
        )
        csv_path: Path = write_gradient_csv(table)
        return _GradientOutcome(
            error=None,
            table=table,
            base_results=base_results,
            base_x=self._base_x,
            mode=self._mode,
            csv_path=csv_path,
            elapsed_seconds=time.perf_counter() - started,
        )


# ==== Objective table ==== #
class _ObjectiveTableModel(QAbstractTableModel):
    """
    A read-only item model over the per-run objective-function
    table, for the panel left of the parameter panel.

    Cells render their values at three decimals with a placeholder
    for a missing or NaN value; the row and column headers are the
    table's own (metric labels and target names).

    Args:
        parent: The parent object; defaults to None.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._table: pd.DataFrame = build_objective_table(
            pd.Series(dtype=float), None
        )

    def set_table(self, table: pd.DataFrame) -> None:
        """
        Replace the underlying table, notifying the view.

        Args:
            table: The new per-run objective table.
        """
        self.beginResetModel()
        self._table = table
        self.endResetModel()

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex = _INVALID_INDEX
    ) -> int:
        """The number of metric rows (zero for any child parent)."""
        return 0 if parent.isValid() else len(self._table)

    def columnCount(
        self, parent: QModelIndex | QPersistentModelIndex = _INVALID_INDEX
    ) -> int:
        """The number of target columns (zero for any child parent)."""
        return 0 if parent.isValid() else len(self._table.columns)

    def index(
        self,
        row: int,
        column: int,
        parent: QModelIndex | QPersistentModelIndex = _INVALID_INDEX,
    ) -> QModelIndex:
        """The index of a cell, or an invalid index for a child parent."""
        if not self.hasIndex(row, column) or parent.isValid():
            return QModelIndex()
        return self.createIndex(row, column)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = 0,
    ) -> str | None:
        """The display text of a cell: the value at three decimals, or a placeholder."""
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        # The table's cells are floats (possibly NaN); the object
        # cells of a mixed-type frame render through their float
        # value.
        cell: Any = self._table.iat[index.row(), index.column()]
        if cell is None or (
            isinstance(cell, (float, np.floating)) and math.isnan(cell)
        ):
            return _MISSING
        return f"{float(cell):.3f}"

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = 0
    ) -> str | None:
        """The column (target) or row (metric) label of a header section."""
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._table.columns[section])
        return str(self._table.index[section])


# ==== Main window ==== #
class MainWindow(QMainWindow):
    """
    The main window of the East River lateral-model calibration GUI.

    The window stacks two non-collapsible splitter sections: the
    figure panel (the model's figures, one visible at a time: the
    three full-mode figures, or the single streamflow figure in
    hydrology-only mode) and, below it, an inner non-collapsible
    splitter with the per-run objective-function table on the left
    and the parameter panel on the right.  The header row carries
    the title, a *Hydrology only* checkbox next to it (off by
    default), a status line, an indeterminate progress bar and
    elapsed-time label (visible only while a run is in flight), a
    *Save run history* checkbox (off by default), an *Auto-run*
    checkbox, and the *Run model*, *Reset bounds*, and *Save run*
    buttons.

    Parameter changes re-run the models on a worker thread, 400 ms
    after the last change, and refresh the figures and the table
    when the run finishes; a generation counter discards stale
    results and starts the next run with the latest values.  With
    the *Auto-run* checkbox unchecked, edits only mark the results
    on screen as stale and the *Run model* button starts the run
    manually.  With the *Hydrology only* checkbox on, a run
    executes the hydrology model alone, scored against the
    measured streamflow: the reactive-transport parameter groups
    and the species / zone / mineral figure checkboxes are hidden,
    the figures reduce to the single streamflow figure, and the
    objective table carries only the streamflow column.  A mode
    change stales the results on screen just like a parameter
    change does.  A run that fails (for instance with
    ``WaterVolumeError``) is reported in the status bar and the
    last valid figures stay on screen.  *Run model* and *Save run*
    are disabled while a run is in flight; *Run model* is also
    disabled while the results on screen already correspond to the
    current parameters.

    At startup the window restores the most recent saved run (its
    parameters, edited bounds, plot selections, run mode, window
    geometry, and splitter sizes) and builds the figures from the
    run's saved output, so no re-run is needed; with no saved run
    it starts from the default parameters and runs once in the
    background.

    Args:
        input_data: The assembled calibration inputs (the
            parameter bounds and the reaction network).
        csv_path: The path of the series CSV the session used, or
            None for the record shipped with the package; stored
            in the saved run state.
        parent: The parent widget; defaults to None.
    """

    def __init__(
        self,
        input_data: GuiInputData,
        csv_path: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._input_data: GuiInputData = input_data
        self._csv_path: str | None = csv_path

        # The last results displayed in the figure and table.
        self._results: GuiRunResults | None = None
        # The run currently in flight (None while no run is in
        # flight; the reference is dropped when the worker ends,
        # before the deferred deletion removes the object).
        self._worker: _RunWorker | _GradientWorker | None = None
        # The generation of the latest started run; an outcome with
        # an older generation is stale and discarded.
        self._run_generation: int = 0
        # The parameter vector of the latest started run (None
        # while no run has started).
        self._start_x: np.ndarray | None = None
        # The run mode of the latest started run (None while no run
        # has started).
        self._start_mode: str | None = None
        # The parameter vector of the latest finished run (None
        # until one has finished): the results on screen correspond
        # to exactly this vector.
        self._last_run_x: np.ndarray | None = None
        # The mode of the latest finished run (None until one has
        # finished): the results on screen were produced in this
        # mode.
        self._last_run_mode: str | None = None
        # The start time of the run in flight, for the elapsed
        # label; None when no run is in flight.
        self._run_started_at: float | None = None
        # The session's previous runs, newest first (capped at
        # _HISTORY_CAP), ahead of the saved run directories the
        # panel's *History* dropdown lists below them.
        self._history_items: list[_HistoryItem] = []
        # The saved runs read from disk, by run directory; a saved
        # entry in _history_items keeps only its path and re-reads
        # its state here (once) when it is restored or overlaid.
        self._saved_run_cache: dict[Path, SavedRun] = {}
        # The last finished gradient job's matrix and metadata
        # (None while no gradient job has finished); the figure
        # panel draws it as an extra page.
        self._gradient_state: _GradientSnapshot | None = None

        self.setWindowTitle(_TITLE)
        self.resize(*_INITIAL_SIZE)

        self._build_ui()
        self._build_actions()
        self._restore_or_start()

    # ==== Building ==== #
    def _build_ui(self) -> None:
        """Assemble the header row and the two-level splitter layout."""
        header: QWidget = self._build_header()

        self._figure_panel: FigurePanel = self._build_figure_panel()
        objective_panel: QWidget = self._build_objective_panel()
        param_panel: ParamPanel = self._build_param_panel()

        # The main splitter stacks the figure over the lower row; the
        # row's own splitter places the objective table left of the
        # parameter panel.
        self._splitter: QSplitter = QSplitter(Qt.Orientation.Vertical)
        self._bottom_splitter: QSplitter = QSplitter(Qt.Orientation.Horizontal)
        self._bottom_splitter.addWidget(objective_panel)
        self._bottom_splitter.addWidget(param_panel)
        self._splitter.addWidget(self._figure_panel)
        self._splitter.addWidget(self._bottom_splitter)
        for section in range(self._splitter.count()):
            self._splitter.setCollapsible(section, False)
        for section in range(self._bottom_splitter.count()):
            self._bottom_splitter.setCollapsible(section, False)
        # Without an explicit distribution Qt lays the sections out
        # from their size hints, which squashes the figure (the
        # canvases report a small hint) into a sliver while the lower
        # row (large hints) takes the rest.
        self._splitter.setSizes(list(_INITIAL_SPLIT_SIZES[:2]))
        self._bottom_splitter.setSizes(list(_INITIAL_SPLIT_SIZES[2:]))
        self._figure_panel.setMinimumHeight(_FIGURE_MIN_HEIGHT)

        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(header)
        layout.addWidget(self._splitter, 1)
        central: QWidget = QWidget(self)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self._debounce_timer: QTimer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._maybe_start_run)

        self._elapsed_timer: QTimer = QTimer(self)
        self._elapsed_timer.setInterval(_ELAPSED_MS)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

    def _build_header(self) -> QWidget:
        """Build the header row: title, status, progress, and buttons."""
        title: QLabel = QLabel(_TITLE)
        font: QFont = QFont(title.font())
        font.setBold(True)
        font.setPointSize(max(9, font.pointSize() + 2))
        title.setFont(font)

        # Off by default: the models run in full mode.
        self._mode_checkbox: QCheckBox = QCheckBox("Hydrology only")
        self._mode_checkbox.setToolTip(
            "Run only the hydrology model, scored against the "
            "measured streamflow.  When checked, the reactive-"
            "transport parameter groups and the species / zone / "
            "mineral figure checkboxes are hidden, the figures "
            "reduce to the single streamflow figure, and the "
            "objective table carries only the streamflow column."
        )
        self._mode_checkbox.toggled.connect(self._on_mode_changed)

        self._status_label: QLabel = QLabel("Preparing…")
        self._status_label.setStyleSheet("color: gray;")

        self._progress_bar: QProgressBar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedWidth(_PROGRESS_WIDTH)
        self._progress_bar.hide()

        self._elapsed_label: QLabel = QLabel("")
        self._elapsed_label.hide()

        self._save_history_checkbox: QCheckBox = QCheckBox(
            "Save run history"
        )
        # Off by default: the history file only grows when the user
        # opts in.
        self._save_history_checkbox.setChecked(False)
        self._save_history_checkbox.setToolTip(
            "Append an abbreviated record of each finished run (the "
            "timestamp, mode, parameter values, and objective "
            "values) to gui_run_history.jsonl in the results "
            "directory."
        )

        self._auto_run_checkbox: QCheckBox = QCheckBox("Auto-run")
        self._auto_run_checkbox.setChecked(True)
        self._auto_run_checkbox.setToolTip(
            "Re-run the models automatically after each parameter "
            "change.  When unchecked, use the Run model button to "
            "start a run manually."
        )

        self._run_button: QPushButton = QPushButton("Run model")
        self._run_button.clicked.connect(self._on_run_clicked)
        self._run_button.setEnabled(False)

        self._gradients_button: QPushButton = QPushButton("Gradients")
        self._gradients_button.clicked.connect(self._on_gradients_clicked)
        self._gradients_button.setToolTip(
            "Differentiate every objective function with respect "
            "to every free parameter by central differences (each "
            "parameter is nudged up and down by 1% of its current "
            "range, clamped to the range's ends).  Every "
            "perturbed point — plus the current results' point "
            "when they are not up to date — is run through the "
            "models, so the job takes about twice the number of "
            "free parameters plus one runs.  The resulting matrix "
            "is written to a CSV in the results directory and "
            "drawn as an extra page of the figure panel."
        )

        self._reset_button: QPushButton = QPushButton("Reset bounds")
        self._reset_button.clicked.connect(self._on_reset_bounds)

        self._save_button: QPushButton = QPushButton("Save run")
        self._save_button.clicked.connect(self._on_save_run)
        self._save_button.setEnabled(False)

        layout: QHBoxLayout = QHBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(title)
        layout.addWidget(self._mode_checkbox)
        layout.addStretch(1)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._elapsed_label)
        layout.addWidget(self._save_history_checkbox)
        layout.addWidget(self._auto_run_checkbox)
        layout.addWidget(self._run_button)
        layout.addWidget(self._gradients_button)
        layout.addWidget(self._reset_button)
        layout.addWidget(self._save_button)
        header: QWidget = QWidget(self)
        header.setLayout(layout)
        return header

    def _build_figure_panel(self) -> FigurePanel:
        """Build the figure panel (its header carries the arrow buttons)."""
        return FigurePanel(self)

    def _build_objective_panel(self) -> QWidget:
        """Build the panel left of the parameter panel: heading, table, and note."""
        self._objective_model: _ObjectiveTableModel = _ObjectiveTableModel(self)
        table: QTableView = QTableView(self)
        table.setModel(self._objective_model)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setShowGrid(False)
        # The row headers (the metric labels) identify what each row
        # measures, so they stay visible.
        table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        heading: QLabel = QLabel("<b>Objective functions</b>")
        note: QLabel = QLabel(METRIC_NOTES)
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")

        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(12, 4, 12, 8)
        layout.setSpacing(4)
        layout.addWidget(heading)
        layout.addWidget(table)
        layout.addWidget(note)
        panel: QWidget = QWidget(self)
        panel.setLayout(layout)
        return panel

    def _build_param_panel(self) -> ParamPanel:
        """Build the parameter panel and connect its signals to the window."""
        self._param_panel: ParamPanel = ParamPanel(
            self._input_data, default_plot_selection(), self
        )
        self._param_panel.value_changed.connect(self._on_value_changed)
        self._param_panel.bounds_changed.connect(self._on_bounds_changed)
        self._param_panel.selection_changed.connect(self._on_selection_changed)
        self._param_panel.history_changed.connect(self._on_history_changed)
        self._param_panel.history_restore_requested.connect(
            self._on_history_restore
        )
        return self._param_panel

    def _build_actions(self) -> None:
        """Register the Left / Right arrow actions that flip the figure pages."""
        previous_action: QAction = QAction("Previous figure", self)
        previous_action.setShortcut(Qt.Key.Key_Left)
        previous_action.triggered.connect(self._figure_panel.previous_page)

        next_action: QAction = QAction("Next figure", self)
        next_action.setShortcut(Qt.Key.Key_Right)
        next_action.triggered.connect(self._figure_panel.next_page)

    # ==== Startup ==== #
    def _restore_or_start(self) -> None:
        """
        Restore the most recent saved run, or start from defaults.

        The parameter panel's *History* dropdown is filled first
        with every saved run directory, newest first, so the
        restored run is one of its entries; with a restorable latest
        run the window then opens on that run's figures and
        parameters without a re-run, and otherwise (no run, or an
        unreadable one) it starts from the default parameters and
        runs the models once in the background.
        """
        runs: list[Path] = list_run_dirs()
        # The saved-run directories become entries of the window's
        # history (label and path only; their state is read on
        # demand), so a selection from the dropdown can restore and
        # overlay the same runs the dropdown shows.
        self._history_items = [
            _HistoryItem(
                label=run.name,
                run_dir=run,
                param_values=None,
                results=None,
                mode=None,
            )
            for run in runs
        ]
        self._param_panel.set_history_items([run.name for run in runs])
        latest: Path | None = runs[0] if runs else None
        if latest is None:
            self._status(
                "No saved run found; starting from the default parameters."
            )
            self._maybe_start_run()
            return
        try:
            saved: SavedRun = load_saved_run(latest, self._input_data)
        except ValueError as err:
            print(
                f"Could not restore the saved run {latest}: {err}",
                file=sys.stderr,
            )
            self._status(
                "The most recent saved run is incomplete; starting from "
                "the default parameters."
            )
            self._maybe_start_run()
            return
        self._apply_saved_run(saved)

    def _apply_saved_run(self, saved: SavedRun) -> None:
        """
        Apply a restored run to the window without a re-run.

        The shared apply path (`_apply_history_entry`) takes the
        saved values and bounds (a degenerate saved bound pins its
        parameter), the checkbox selection, the saved run mode, and
        the run's saved results, and marks them as the last run so
        *Save run* is immediately available; this method then adds
        the saved-window bits: the status note, the window geometry
        and splitter sizes, and the *Save run history* checkbox.
        """
        self._apply_history_entry(
            saved.param_values,
            saved.param_bounds,
            saved.selection,
            saved.results,
            saved.mode,
        )
        if saved.restore_note is not None:
            self._status(saved.restore_note)
        else:
            self._status(
                "Restored the most recent saved run; the figures were "
                "built from its saved output."
            )
        if saved.window_geometry is not None:
            self.setGeometry(
                saved.window_geometry.get("x", 0),
                saved.window_geometry.get("y", 0),
                saved.window_geometry.get("width", _INITIAL_SIZE[0]),
                saved.window_geometry.get("height", _INITIAL_SIZE[1]),
            )
        if saved.splitter_sizes is not None:
            self._splitter.setSizes(saved.splitter_sizes[:2])
            self._bottom_splitter.setSizes(saved.splitter_sizes[2:])
        # The checkbox state is part of the saved state; a state
        # saved before the toggle existed restores it off.
        self._save_history_checkbox.setChecked(saved.save_history)

    # ==== Runs ==== #
    def _current_x(self) -> np.ndarray:
        """
        The current parameter vector, in the loaded bounds' key order.

        Returns:
            The values of every parameter as a float64 array.
        """
        values: dict[str, float] = self._param_panel.param_values()
        return np.array(
            [values[name] for name in self._input_data.bounds], dtype=np.float64
        )

    def _current_mode(self) -> str:
        """
        The run mode the next run will use.

        Returns:
            `MODE_HYDRO` when the header's *Hydrology only*
            checkbox is on, `MODE_FULL` otherwise.
        """
        return MODE_HYDRO if self._mode_checkbox.isChecked() else MODE_FULL

    def _results_are_up_to_date(self) -> bool:
        """
        Whether the results on screen correspond to the current
        parameters and run mode.

        The figures and the table show the last finished run's
        results; they are up to date when that run used the
        parameters the panel holds now, in the current run mode.
        """
        return (
            self._results is not None
            and self._last_run_x is not None
            and self._last_run_mode is not None
            and self._last_run_mode == self._current_mode()
            and np.array_equal(self._current_x(), self._last_run_x)
        )

    def _update_run_button(self) -> None:
        """
        Sync the run button's enabled state with the window's.

        The button is the manual trigger: it is enabled when no run
        is in flight and the results on screen do not correspond to
        the current parameters, that is, exactly when a manual run
        would produce new results.
        """
        self._run_button.setEnabled(
            self._worker is None and not self._results_are_up_to_date()
        )

    def _maybe_start_run(self) -> None:
        """
        Start a run with the current parameters, unless there is
        nothing to do.

        Skipped while a run is in flight, when the results on
        screen already correspond to the current parameters (no
        change since the last finished run), and when auto-run is
        off (the run button is the manual trigger then; the check
        here also covers a debounce that outlives the checkbox being
        turned off).
        """
        if not self._auto_run_checkbox.isChecked():
            return
        if self._worker is not None and self._worker.isRunning():
            return
        if self._results_are_up_to_date():
            return
        self._start_run(self._current_x())

    def _on_run_clicked(self) -> None:
        """
        The run model button: start a run with the current
        parameters.

        Bypasses the auto-run checkbox (the user asked for this run
        explicitly); skipped while a run is in flight or when the
        results on screen already correspond to the current
        parameters (the button is disabled in both cases, so this
        only guards a click that races a run start).
        """
        if self._worker is not None and self._worker.isRunning():
            return
        if self._results_are_up_to_date():
            return
        self._start_run(self._current_x())

    def _start_run(self, x: np.ndarray) -> None:
        """Start a worker run with `x` and show the run indicators."""
        self._run_generation += 1
        mode: str = self._current_mode()
        self._start_x = x
        self._start_mode = mode
        self._run_started_at = time.perf_counter()
        self._progress_bar.show()
        self._elapsed_label.show()
        self._run_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._gradients_button.setEnabled(False)
        self._status("Running the model…")

        worker: _RunWorker = _RunWorker(
            self._run_generation, x, self._input_data, mode, self
        )
        worker.progress_line.connect(self._on_progress_line)
        worker.finished_run.connect(self._on_finished_run)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        self._elapsed_timer.start()
        worker.start()

    def _on_progress_line(self, line: str) -> None:
        """Show the run's latest verbose line in the status bar."""
        if line:
            self._status(f"Running the model — {line}")

    def _update_elapsed(self) -> None:
        """Refresh the elapsed-time label for the run in flight."""
        if self._run_started_at is None:
            return
        seconds: int = int(time.perf_counter() - self._run_started_at)
        minutes, secs = divmod(seconds, 60)
        self._elapsed_label.setText(f"{minutes:02d}:{secs:02d}")

    def _on_worker_finished(self) -> None:
        """
        The worker thread has ended: drop the window's reference.

        The reference is cleared before the deferred deletion
        (``deleteLater``, connected after this handler) runs, so a
        later ``isRunning`` check never touches a deleted C++
        object.
        """
        self._worker = None

    def _on_finished_run(self, payload: object) -> None:
        """
        Apply a worker outcome to the window.

        An outcome whose generation no longer matches the latest
        started run is stale (the parameters or the run mode
        changed mid-run) and is discarded; the window then starts a
        fresh run with the latest values.  A failed run is reported
        in the status bar and the last valid figures stay on
        screen.
        """
        self._elapsed_timer.stop()
        self._progress_bar.hide()
        self._elapsed_label.hide()

        if not isinstance(payload, _RunOutcome):
            return
        outcome: _RunOutcome = payload
        if (
            outcome.generation != self._run_generation
            or self._start_x is None
            or self._start_mode is None
        ):
            return
        completed_x: np.ndarray = self._start_x
        completed_mode: str = self._start_mode
        self._start_x = None
        self._start_mode = None
        self._run_started_at = None

        if outcome.error is not None:
            self._status(f"Run failed: {outcome.error}")
        elif outcome.results is None:
            self._status("Run failed: the worker reported no results.")
        else:
            self._results = outcome.results
            self._last_run_x = completed_x
            self._last_run_mode = completed_mode
            self._status(
                f"Run {self._run_generation} finished in "
                f"{outcome.elapsed_seconds:.1f} s."
            )
            self._append_run_history()
            self._record_run_history()
            self._refresh_from_results()

        # The results on screen correspond to this run's parameters.
        x_up_to_date: bool = (
            self._last_run_x is not None
            and np.array_equal(completed_x, self._last_run_x)
        )
        # …and this run used the mode the window is in now; a mode
        # change leaves the old results stale even when the
        # parameters are unchanged.
        up_to_date: bool = (
            x_up_to_date and completed_mode == self._current_mode()
        )
        self._save_button.setEnabled(up_to_date)
        # The run button mirrors the same staleness; it does not look
        # at the worker here because this handler runs before the
        # worker's finished signal clears the reference.
        self._run_button.setEnabled(not up_to_date)
        # No run is in flight any more, so the gradient job can
        # start again (a stale restart below re-disables it via
        # `_start_run`).
        self._gradients_button.setEnabled(True)
        # The parameters or the mode changed while this run was in
        # flight: the results are stale, so start the next one with
        # the latest values (this is what the generation counter is
        # for), unless the user has auto-run switched off.
        if (
            x_up_to_date
            and (
                not np.array_equal(completed_x, self._current_x())
                or completed_mode != self._current_mode()
            )
            and self._auto_run_checkbox.isChecked()
        ):
            self._start_run(self._current_x())

    # ==== Gradients ==== #
    def _on_gradients_clicked(self) -> None:
        """
        Start a gradient job from the header's *Gradients* button.

        A job already in flight (a model run or an earlier
        gradient job) takes precedence: the click is ignored,
        since a job cannot be interrupted.
        """
        if self._worker is not None and self._worker.isRunning():
            return
        self._start_gradient()

    def _start_gradient(self) -> None:
        """
        Start a finite-difference gradient job around the current
        parameters.

        Every free parameter (a parameter whose current range has
        a positive width) is perturbed up and down by 1% of its
        range and each perturbed point — plus the base point when
        the results on screen are not up to date — is run through
        the models on a worker thread in the current run mode.
        The run indicators show the job's progress like a model
        run's; a job cannot be interrupted.
        """
        x: np.ndarray = self._current_x()
        steps: list[_Perturbation] = free_parameter_steps(
            x, self._param_panel.current_bounds()
        )
        if not steps:
            self._status("No free parameters to differentiate.")
            return
        base_results: GuiRunResults | None = (
            self._results
            if self._results is not None and self._results_are_up_to_date()
            else None
        )

        self._run_started_at = time.perf_counter()
        self._progress_bar.show()
        self._elapsed_label.show()
        self._run_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._gradients_button.setEnabled(False)
        self._status("Computing gradients…")

        worker: _GradientWorker = _GradientWorker(
            x,
            steps,
            self._current_mode(),
            self._input_data,
            base_results,
            self,
        )
        worker.progress_line.connect(self._on_gradient_progress_line)
        worker.finished_gradient.connect(self._on_finished_gradient)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        self._elapsed_timer.start()
        worker.start()

    def _on_gradient_progress_line(self, line: str) -> None:
        """Show the gradient job's latest progress line in the status bar."""
        if line:
            self._status(f"Computing gradients — {line}")

    def _on_finished_gradient(self, payload: object) -> None:
        """
        Apply a finished gradient job's outcome to the window.

        A failed job is reported in the status bar and leaves the
        figures on screen unchanged.  A finished job's matrix is
        kept as the figure panel's extra page, its base point's
        results are adopted as the window's current results when
        they match the window's current parameters and mode (so
        the run and the save button light up as after a normal
        run), and its completion is reported in the status bar
        with the path of the written CSV.
        """
        self._elapsed_timer.stop()
        self._progress_bar.hide()
        self._elapsed_label.hide()
        self._run_started_at = None

        if not isinstance(payload, _GradientOutcome):
            return
        outcome: _GradientOutcome = payload
        if outcome.error is not None:
            self._status(f"Gradients failed: {outcome.error}")
        elif outcome.table is None or outcome.csv_path is None:
            self._status("Gradients failed: the worker reported no result.")
        else:
            table: pd.DataFrame = outcome.table
            csv_path: Path = outcome.csv_path
            if (
                outcome.base_results is not None
                and outcome.mode == self._current_mode()
                and np.array_equal(outcome.base_x, self._current_x())
            ):
                self._results = outcome.base_results
                self._last_run_x = outcome.base_x
                self._last_run_mode = outcome.mode
            self._gradient_state = _GradientSnapshot(
                table=table,
                mode=outcome.mode,
                csv_path=csv_path,
                elapsed_seconds=outcome.elapsed_seconds,
            )
            self._refresh_from_results()
            self._status(
                f"Gradients finished in {outcome.elapsed_seconds:.1f} s — "
                f"saved to {csv_path}."
            )

        # No job is in flight by the time this handler runs (the
        # worker's finished signal clears the reference after this
        # one, like in `_on_finished_run`): the run and the save
        # buttons mirror the results' staleness and the gradients
        # button always comes back up.
        up_to_date: bool = self._results_are_up_to_date()
        self._save_button.setEnabled(up_to_date)
        self._run_button.setEnabled(not up_to_date)
        self._gradients_button.setEnabled(True)

    def _record_run_history(self) -> None:
        """
        Append the finished run to the session's run-history file.

        With the *Save run history* checkbox on, one abbreviated
        record (timestamp, mode, parameter values, and objective
        values) is appended to the results directory's
        ``gui_run_history.jsonl``; with it off nothing is written.
        Recording the history never disturbs the results on screen
        (a write failure only prints to the standard error, in
        `append_run_history`).
        """
        if (
            not self._save_history_checkbox.isChecked()
            or self._results is None
        ):
            return
        # Record the mode of the results on screen: the last
        # finished run's (set when it completed), or the current
        # one if this is ever called before any run has finished.
        mode: str = self._last_run_mode or self._current_mode()
        entry: RunHistoryEntry = RunHistoryEntry(
            timestamp=datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            mode=mode,
            param_values=self._param_panel.param_values(),
            objectives=objectives_from_results(self._results),
        )
        append_run_history(Path("results"), entry)

    # ==== History ==== #
    def _append_run_history(self) -> None:
        """
        Record a finished session run in the window's history.

        The run's parameters, results, and mode become a new entry
        at the front of the session's part of the *History* list
        (ahead of the older session entries and the saved run
        directories, which keep their own order below it); the
        oldest session entry is dropped once the session's part of
        the list reaches _HISTORY_CAP, so a long session never
        bloats the dropdown.
        """
        if self._results is None or self._last_run_mode is None:
            return
        mode: str = self._last_run_mode
        label: str = (
            f"Run {self._run_generation} — "
            f"{datetime.now().strftime('%H:%M:%S')} "
            f"({'full' if mode == MODE_FULL else 'hydro'})"
        )
        item: _HistoryItem = _HistoryItem(
            label=label,
            run_dir=None,
            param_values=self._param_panel.param_values(),
            results=self._results,
            mode=mode,
        )
        session_entries: list[_HistoryItem] = [
            entry for entry in self._history_items if entry.run_dir is None
        ]
        session_entries.insert(0, item)
        del session_entries[_HISTORY_CAP:]
        saved_entries: list[_HistoryItem] = [
            entry for entry in self._history_items if entry.run_dir is not None
        ]
        self._history_items = session_entries + saved_entries
        self._sync_history_combo()

    def _add_saved_history_item(self, run_dir: Path) -> None:
        """
        Add a just-saved run directory to the *History* list.

        A directory the list already carries (the same run saved
        again) is not duplicated; its refreshed entry goes to the
        front of the saved-run part of the list, so the newest save
        appears first, right below the session's runs, and any
        previously read state of the directory is dropped from the
        cache.  The entry keeps only the directory's path and
        label; its state is read on demand.
        """
        if run_dir in self._saved_run_cache:
            del self._saved_run_cache[run_dir]
        session_entries: list[_HistoryItem] = [
            entry for entry in self._history_items if entry.run_dir is None
        ]
        saved_entries: list[_HistoryItem] = [
            entry
            for entry in self._history_items
            if entry.run_dir is not None and entry.run_dir != run_dir
        ]
        saved_entries.insert(
            0,
            _HistoryItem(
                label=run_dir.name,
                run_dir=run_dir,
                param_values=None,
                results=None,
                mode=None,
            ),
        )
        self._history_items = session_entries + saved_entries
        self._sync_history_combo()

    def _sync_history_combo(self) -> None:
        """
        Repaint the *History* dropdown from the window's history.

        The items are the history's labels in list order (the
        session's runs, newest first, then the saved run
        directories, newest first); the panel's own method blocks
        its signals while it works, so the repaint does not
        re-emit `history_changed`.
        """
        self._param_panel.set_history_items(
            [item.label for item in self._history_items]
        )

    def _selected_history_item(self) -> _HistoryItem | None:
        """
        The history entry selected in the *History* dropdown.

        Returns:
            The entry, or None when the dropdown is empty.
        """
        index: int = self._param_panel.history_selected_index()
        if index < 0 or index >= len(self._history_items):
            return None
        return self._history_items[index]

    def _load_saved_history_item(
        self, item: _HistoryItem
    ) -> SavedRun | None:
        """
        Read a saved-run entry's state from disk, once.

        The entry's directory is read through `load_saved_run` and
        kept in the window's cache, so a second restore or overlay
        of the same run does not re-read it; a directory that has
        gone away, or whose state no longer reads, yields None, a
        status note, and a line on the standard error.

        Args:
            item: The saved-run history entry (with a run_dir).

        Returns:
            The read `SavedRun`, or None when the run can no
            longer be read.
        """
        assert item.run_dir is not None
        if item.run_dir in self._saved_run_cache:
            return self._saved_run_cache[item.run_dir]
        try:
            saved: SavedRun = load_saved_run(item.run_dir, self._input_data)
        except ValueError as err:
            print(
                f"Could not restore the saved run {item.run_dir}: {err}",
                file=sys.stderr,
            )
            self._status(
                f"The previous run {item.label} can no longer be restored."
            )
            return None
        self._saved_run_cache[item.run_dir] = saved
        return saved

    def _selected_overlay(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None] | None:
        """
        The previous run's frames for the figures' overlay, when
        asked for.

        The overlay is drawn when the *Overlay previous results*
        checkbox is on and a previous run is selected in the
        dropdown: the frames are that run's saved (or in-memory)
        simulation DataFrames, clipped by the figures themselves to
        the plotted window.  A session run's frames come straight
        from its stored results; a saved run's are read (once) from
        its run directory.

        Returns:
            The previous run's frames as (hydrology simulation,
            reactive-transport simulation), or None when the
            overlay is off, no run is selected, or a saved run's
            directory can no longer be read.
        """
        if not self._param_panel.history_overlay_enabled():
            return None
        item: _HistoryItem | None = self._selected_history_item()
        if item is None:
            return None
        results: GuiRunResults
        if item.run_dir is not None:
            saved: SavedRun | None = self._load_saved_history_item(item)
            if saved is None:
                return None
            results = saved.results
        else:
            assert item.results is not None
            results = item.results
        rt_sim: pd.DataFrame | None = (
            results.rt.simulation if results.rt is not None else None
        )
        return (results.hydro.simulation, rt_sim)

    def _on_history_changed(self) -> None:
        """
        Answer a change in the *History* dropdown or the overlay
        checkbox.

        Neither control re-runs the models: the figures are simply
        rebuilt from the last results, the previous run's overlay
        reapplied or dropped as the checkbox asks.
        """
        self._refresh_from_results()

    def _on_history_restore(self) -> None:
        """
        Restore the selected previous run to the panel, without a
        re-run.

        A session run's values, results, and mode come straight
        from its stored entry (its bounds and plot selection keep
        the panel's current ones); a saved run's are read from its
        run directory.  The panel's values, mode, and the results
        on screen are replaced by the entry's, and the run button
        stays disabled — the restored results correspond to the
        restored parameters.

        Status:
            A note naming the restored run, or an explanatory note
            when the dropdown is empty.
        """
        item: _HistoryItem | None = self._selected_history_item()
        if item is None:
            self._status("No previous run selected: the history is empty.")
            return
        if item.run_dir is not None:
            saved: SavedRun | None = self._load_saved_history_item(item)
            if saved is None:
                return
            self._apply_history_entry(
                saved.param_values,
                saved.param_bounds,
                saved.selection,
                saved.results,
                saved.mode,
            )
        else:
            assert item.param_values is not None
            assert item.results is not None
            assert item.mode is not None
            self._apply_history_entry(
                item.param_values,
                None,
                None,
                item.results,
                item.mode,
            )
        self._status(f"Restored the previous run {item.label}.")

    def _apply_history_entry(
        self,
        values: dict[str, float],
        bounds: dict[str, tuple[float, float]] | None,
        selection: PlotSelection | None,
        results: GuiRunResults,
        mode: str,
    ) -> None:
        """
        Apply a history entry to the window without a re-run.

        The panel takes the entry's parameter values and, when
        given, its bounds (a degenerate bound pins its parameter);
        a session entry passes None for both and keeps the
        panel's current ones.  When given, the entry's plot
        selection replaces the panel's checkbox state (the panel
        methods emit no signals, so the restore triggers no
        re-run).  The window marks the entry's results as the last
        run's, in the entry's mode, so *Save run* is immediately
        available and the run button stays disabled until a
        parameter changes.

        Args:
            values: The entry's parameter values, by key.
            bounds: The entry's parameter bounds, by key; None
                keeps the panel's current bounds.
            selection: The entry's plot selection; None keeps the
                panel's current one.
            results: The entry's results (the figures and the
                table are rebuilt from them).
            mode: The entry's run mode.
        """
        current_bounds: dict[str, tuple[float, float]] = (
            self._param_panel.current_bounds()
        )
        self._param_panel.restore_state(
            values, bounds if bounds is not None else current_bounds
        )
        if selection is not None:
            self._param_panel.set_selection(selection)
        self._results = results
        self._last_run_x = self._current_x()
        # The run mode is part of the entry too: the checkbox and
        # the panel's visibility follow it (the checkbox's signal
        # is blocked, so the restore does not trigger a re-run).
        self._mode_checkbox.blockSignals(True)
        self._mode_checkbox.setChecked(mode == MODE_HYDRO)
        self._mode_checkbox.blockSignals(False)
        self._param_panel.set_hydro_only(mode == MODE_HYDRO)
        self._last_run_mode = mode
        self._save_button.setEnabled(True)
        # The restored results correspond to the restored parameters,
        # so there is nothing for a manual run to do yet.
        self._update_run_button()
        self._refresh_from_results()

    # ==== Parameter panel ==== #
    def _on_mode_changed(self, _enabled: bool) -> None:
        """
        The header's *Hydrology only* checkbox was toggled.

        The parameter panel hides or shows its reactive-transport
        parts, the results on screen are stale in the new mode (the
        run button reflects that), and with auto-run on a debounced
        re-run follows so the figures match the mode.
        """
        mode: str = self._current_mode()
        self._param_panel.set_hydro_only(mode == MODE_HYDRO)
        self._update_run_button()
        if self._auto_run_checkbox.isChecked():
            self._debounce_timer.start()

    def _on_value_changed(self, _key: str) -> None:
        """
        A parameter value changed: re-run after the debounce, when
        auto-run is on.

        The run button always reflects the new staleness of the
        results on screen, so it can start the run manually.
        """
        self._update_run_button()
        if self._auto_run_checkbox.isChecked():
            self._debounce_timer.start()

    def _on_bounds_changed(self, _key: str, _lo: float, _hi: float) -> None:
        """
        A parameter range changed (the value changed with it):
        re-run after the debounce, when auto-run is on.
        """
        self._update_run_button()
        if self._auto_run_checkbox.isChecked():
            self._debounce_timer.start()

    def _on_selection_changed(self, _selection: PlotSelection) -> None:
        """
        The plot selection changed: rebuild the figures from the
        last results without re-running the models.
        """
        self._refresh_from_results()

    def _on_reset_bounds(self) -> None:
        """Confirm and reset all parameter ranges to their loaded defaults."""
        answer: QMessageBox.StandardButton = QMessageBox.question(
            self,
            "Reset bounds",
            "Reset all parameter ranges to their loaded defaults? The "
            "current values are clamped into the restored ranges.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._param_panel.reset_bounds()
        self._status("Parameter ranges reset to their loaded defaults.")
        self._update_run_button()
        if self._auto_run_checkbox.isChecked():
            self._debounce_timer.start()

    # ==== Save ==== #
    def _on_save_run(self) -> None:
        """
        Save the current run: the existing per-run model output plus
        a ``gui_state.json`` capturing the session state.

        The run is saved only when the results on screen correspond
        to the current parameters and mode (a run still catching up
        blocks the save, since its output would be stale).  A
        full-mode run saves the ``hydro`` and ``rt`` bundle pair
        and a hydrology-only run the flat hydrology bundle; the
        state file records the run mode.  The new run directory
        becomes the first entry in the panel's *History* group (the
        newest save), so it can be restored from the dropdown
        without a re-run.
        """
        if self._results is None:
            self._status("Nothing to save yet: the model has not run.")
            return
        if not self._results_are_up_to_date():
            self._status(
                "The model is still catching up; let the current run "
                "finish before saving."
            )
            return

        # The run saver needs the model that produced the results
        # (for the saved parameter table and the run metadata); it
        # keeps no reference to that model, so it is rebuilt from
        # the same parameter array: arr_to_model is construction
        # only, with the same defaults as the run's arr_to_results
        # call, so it is the exact model that produced this run.
        rt_results: pt.RtModelResults | None = self._results.rt
        if rt_results is None:
            # A hydrology-only run saves one flat bundle
            # (simulation, parameters, and objective functions
            # directly in the run directory).
            num_hydro: int = (
                pt.LateralThreeLayerModel.get_num_hydro_parameters()
            )
            hydro_model: HydrologicalModel = (
                pt.LateralThreeLayerModel.hydro_from_array(
                    self._current_x()[:num_hydro], natural_scales=True
                )
            )
            run_dir: Path = save_hydro_run(
                hydro_model, self._results.hydro, results_dir=None
            )
        else:
            model: pt.Model
            _initial_state: np.ndarray
            model, _initial_state = arr_to_model(
                self._current_x(), self._input_data
            )
            run_dir = save_model_run(
                model,
                self._results.hydro,
                rt_results,
                results_dir=None,
            )

        # The client geometry (rather than the frame's): the restore
        # path applies these values with setGeometry, so saving the
        # frame size would grow the window on every save and restore.
        geom: QRect = self.geometry()
        state: GuiState = GuiState(
            csv_path=self._csv_path,
            param_values=self._param_panel.param_values(),
            param_bounds=self._param_panel.current_bounds(),
            selection=self._param_panel.selection(),
            window_geometry={
                "x": geom.x(),
                "y": geom.y(),
                "width": geom.width(),
                "height": geom.height(),
            },
            splitter_sizes=[
                int(size)
                for size in [
                    *self._splitter.sizes(),
                    *self._bottom_splitter.sizes(),
                ]
            ],
            save_history=self._save_history_checkbox.isChecked(),
            mode=self._current_mode(),
        )
        write_gui_state(run_dir, state)
        self._add_saved_history_item(run_dir)
        self._status(f"Saved run to {run_dir}.")

    # ==== Refresh ==== #
    def _refresh_from_results(self) -> None:
        """
        Rebuild the figures and the objective table from the last
        results.

        Every figure honors the current plot selection, so a
        selection change refreshes them without a re-run.  A
        hydrology-only run's results carry no reactive-transport
        simulation, so only the single streamflow figure is shown
        and the objective table carries only the streamflow column.
        A previous run selected in the panel's *History* group
        (with its overlay checkbox on) is drawn faintly on every
        figure, behind the current run's series.  When a gradient
        job has finished, its difference matrix is drawn as an
        extra page after the mode's own figures.
        """
        if self._results is None:
            return
        hydro: pt.HydroModelResults = self._results.hydro
        rt: pt.RtModelResults | None = self._results.rt
        selection: PlotSelection = self._param_panel.selection()

        # The previous run selected in the *History* group (or None
        # when the overlay is off or no run is selected), drawn
        # faintly on every figure.
        overlay: tuple[pd.DataFrame, pd.DataFrame | None] | None = (
            self._selected_overlay()
        )
        overlay_hydro: pd.DataFrame | None = (
            overlay[0] if overlay is not None else None
        )
        overlay_rt: pd.DataFrame | None = (
            overlay[1] if overlay is not None else None
        )

        hydro_sim: pd.DataFrame = hydro.simulation
        if rt is None:
            # A hydrology-only run shows the streamflow figure
            # alone, with no reactive-transport column in the
            # table.
            time_series: Figure = make_time_series_figure(
                hydro_sim,
                None,
                self._input_data.q_meas,
                self._input_data.conc_df,
                selection.time_series,
                overlay_hydro_sim=overlay_hydro,
                overlay_rt_sim=overlay_rt,
            )
            figures: tuple[tuple[str, Figure], ...] = (
                ("Streamflow", time_series),
            )
            table: pd.DataFrame = build_objective_table(
                hydro.objective_functions, None
            ).round(3)
        else:
            rt_sim: pd.DataFrame = rt.simulation
            time_series = make_time_series_figure(
                hydro_sim,
                rt_sim,
                self._input_data.q_meas,
                self._input_data.conc_df,
                selection.time_series,
                overlay_hydro_sim=overlay_hydro,
                overlay_rt_sim=overlay_rt,
            )
            zone: Figure = make_zone_figure(
                rt_sim,
                hydro_sim,
                selection.species,
                selection.zones,
                overlay_rt_sim=overlay_rt,
                overlay_hydro_sim=overlay_hydro,
            )
            rate: Figure = make_rate_figure(
                rt_sim, selection.minerals, overlay_rt_sim=overlay_rt
            )
            figures = (
                ("Streamflow and river concentrations", time_series),
                ("Zone concentrations and storage", zone),
                ("Mineral reaction rates", rate),
            )
            table = build_objective_table(
                hydro.objective_functions, rt.objective_functions
            ).round(3)

        # The last finished gradient job's matrix rides along as an
        # extra page, so it stays on screen across runs, mode
        # changes, and plot-selection changes until the next
        # successful job replaces it.
        if self._gradient_state is not None:
            snapshot: _GradientSnapshot = self._gradient_state
            note: str = (
                "Central differences, h = 1% of each parameter's "
                f"range; saved to {snapshot.csv_path}."
            )
            mode_label: str = "hydro" if snapshot.mode == MODE_HYDRO else "full"
            gradient_fig: Figure = make_gradient_figure(snapshot.table, note)
            figures = figures + ((f"Gradients ({mode_label})", gradient_fig),)

        self._figure_panel.set_figures(figures)
        self._objective_model.set_table(table)

    # ==== Status ==== #
    def _status(self, text: str) -> None:
        """Set the status line of the header row."""
        self._status_label.setText(text)

    # ==== Closing ==== #
    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Close the window, waiting for a job in flight to finish.

        Jobs cannot be interrupted: when a job is in progress (a
        model run or a gradient job) the user is asked whether to
        wait, and the event loop pumps until the worker finishes
        (the window stays responsive and shows the final results)
        or the close is cancelled.
        """
        if self._worker is not None and self._worker.isRunning():
            answer: QMessageBox.StandardButton = QMessageBox.question(
                self,
                "Model run in progress",
                "A model run is in progress and cannot be interrupted. "
                "Wait for it to finish before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            while self._worker is not None and self._worker.isRunning():
                QCoreApplication.processEvents()
                time.sleep(0.05)
        super().closeEvent(event)
