"""
The main window of the East River lateral-model calibration GUI.

The window stacks two sections: the figure panel (the model's
figures, one visible at a time, flipped by the panel's arrow
buttons or the main window's Left / Right keys: the three
full-mode figures, or the streamflow and zone-storage figures in
hydrology-only mode, plus the gradient job's matrix and a line
search's value profile as extra pages when those jobs have
finished) and, below it, a row holding the fixed-size per-run
objective-function table on the left and the parameter panel on
the right.

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
groups are hidden, the figures reduce to the streamflow figure
and a figure of each zone's water storage, and the objective
table carries only the streamflow column.

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

The header's *Line search* button follows the steepest ascent of
one objective function — chosen from a menu listing the
objectives the current results report a finite value for, among
the metrics with 1 as their optimal value — from the current
parameters: the direction comes from the last gradient job's
matrix when one has run at these parameters, or from the same
central differences otherwise, and the objective is then
evaluated at 25 points on the ascent line, all inside the current
parameter bounds.  The best point's parameters are adopted into
the panel, and the value profile is written to a
``line_search_YYYYMMDD_HHMMSS.csv`` file in the results directory
and drawn as an extra page of the figure panel.

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

A *Save run* click writes the existing per-run model output, the
figures as plotted (PNGs into the run directory's ``figures``
subfolder), and a ``gui_state.json`` capturing the session state,
so the next start restores exactly what the user saw.  With the
header's *Save run history* checkbox on, every finished run also
appends one abbreviated record (timestamp, mode, parameter values,
objective values) to a ``gui_run_history.jsonl`` file in the
results directory.
"""

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from io import TextIOBase
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import potions as pt
from matplotlib.figure import Figure
from potions.hydro_model import HydrologicalModel
from potions.reactive_transport.rt_zone import WaterVolumeError
from potions.result_saving import save_hydro_run, save_model_run
from PySide6.QtCore import (
    QAbstractTableModel,
    QCoreApplication,
    QObject,
    QModelIndex,
    QPersistentModelIndex,
    QRect,
    QSize,
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
    QMenu,
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
    DEFAULT_CSV_PATH,
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
from potions.gui.line_search import (
    expand_direction,
    feasible_t_interval,
    write_line_search_csv,
)
from potions.gui.model_pool import map_model_runs
from potions.gui.param_panel import ParamPanel
from potions.gui.plots import (
    build_objective_table,
    make_gradient_figure,
    make_line_search_figure,
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

# The initial window size (px); a restored window geometry replaces
# it before the window is shown.
_INITIAL_SIZE: Final[tuple[int, int]] = (1200, 900)

# The initial sizes (px) of the main vertical splitter: the figure
# and the lower (table-and-parameters) row.  Qt scales the pair
# proportionally to the splitter's size on the first layout, and a
# restored session state replaces them with its own.
_INITIAL_SPLIT_SIZES: Final[tuple[int, int]] = (350, 530)

# The status label's width limits (px): an unbounded label would
# grow the window's minimum width with every status message, so the
# label reports a small constant minimum hint (its text is clipped
# when the window is narrow; the full text stays in the tooltip)
# and is capped at the maximum width.
_STATUS_MIN_WIDTH: Final[int] = 80
_STATUS_MAX_WIDTH: Final[int] = 350

# The side (px) of the header's running-indicator square.  Its
# fixed size keeps the header's — and so the window's — minimum
# width constant whether or not a job is running.
_SPINNER_SIZE: Final[int] = 24

# The period (ms) of the running indicator's frame-advance timer.
_SPINNER_INTERVAL_MS: Final[int] = 100

# The running indicator's frames, in cycle order.
_SPINNER_FRAMES: Final[tuple[str, ...]] = ("▖", "▘", "▝", "▗")

# The fixed objective panel's contents margins (px) in the order
# left, top, right, bottom: the panel is sized exactly to fit its
# heading and table plus these margins, so they are the only
# padding around them.
_OBJECTIVE_PANEL_MARGINS: Final[tuple[int, int, int, int]] = (12, 4, 12, 8)

# The vertical spacing (px) between the fixed objective panel's
# heading and its table.
_OBJECTIVE_PANEL_SPACING: Final[int] = 4

# The number of points the line search evaluates on the ascent
# line, between the feasible interval's ends.
_LINE_SEARCH_POINTS: Final[int] = 25

# The metrics the line search can optimize: the metrics with 1 as
# their optimal value, so the search maximizes them (minimizing
# ``1 - value`` would find the same best point).
_LINE_SEARCH_METRICS: Final[frozenset[str]] = frozenset(
    {"KGE", "LogKGE", "NSE", "LogNSE"}
)

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
def _objective_panel_size(
    heading_size: QSize,
    column_widths: list[int],
    row_heights: list[int],
    vertical_header_width: int,
    horizontal_header_height: int,
    frame_width: int,
    margins: tuple[int, int, int, int],
    spacing: int,
) -> tuple[int, int]:
    """
    The fixed objective panel's size that exactly fits its contents.

    The width is the wider of the heading's hint and the table's
    full width (the row-header column, every column section, and
    the frame on both sides); the height stacks the heading, the
    table (the column-header row, every row section, and the
    frame), the panel's contents margins, and the heading-to-table
    spacing.  Holding the panel at exactly this size means neither
    scrollbar can ever be needed.

    Args:
        heading_size: The heading label's size hint.
        column_widths: The table's per-column section size hints.
        row_heights: The table's per-row section size hints.
        vertical_header_width: The row-header column's width.
        horizontal_header_height: The column-header row's height.
        frame_width: The table view's frame width, per side.
        margins: The panel's contents margins (left, top, right,
            bottom).
        spacing: The panel's layout spacing between the heading
            and the table.

    Returns:
        The panel's (width, height).
    """
    table_w: int = vertical_header_width + sum(column_widths) + 2 * frame_width
    table_h: int = horizontal_header_height + sum(row_heights) + 2 * frame_width
    panel_w: int = max(heading_size.width(), table_w) + margins[0] + margins[2]
    panel_h: int = heading_size.height() + table_h + margins[1] + margins[3] + spacing
    return (panel_w, panel_h)


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
        exception: The failure's exception, or None on success; the
            window shows a water-volume failure as a modal dialog
            and keeps the exception for that.
    """

    generation: int
    results: GuiRunResults | None
    error: str | None
    elapsed_seconds: float
    exception: Exception | None


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
        progress_step: Emitted with (done, total) step counts:
            (0, 1) before the models run and (1, 1) before the
            outcome, on both the success and the error path, for
            the window's progress bar.

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
    progress_step = Signal(int, int)

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
        exception: Exception | None = None
        self.progress_step.emit(0, 1)
        try:
            if self._mode == MODE_HYDRO:
                results = arr_to_hydro_results(self._x, self._input_data)
            else:
                results = arr_to_results(self._x, self._input_data)
        except Exception as err:
            results = None
            exception = err
            error = f"{type(err).__name__}: {err}"
        finally:
            sys.stdout = real_stdout
        self.progress_step.emit(1, 1)
        self.finished_run.emit(
            _RunOutcome(
                self._generation,
                results,
                error,
                time.perf_counter() - started,
                exception,
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

    A non-base point that failed does not fail the job: its column
    of the matrix is all NaN (the missing value), and the counts
    of the failed and the evaluated points ride on the outcome so
    the window can report them in the completion status.

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
        failed_points: The number of the job's non-base points
            that failed: their matrix columns are all NaN.  Always
            zero on a failed job.
        total_points: The number of points the job evaluated (the
            deduplicated perturbed points, the base included when
            it ran).  Always zero on a failed job.
    """

    error: str | None
    table: pd.DataFrame | None
    base_results: GuiRunResults | None
    base_x: np.ndarray
    mode: str
    csv_path: Path | None
    elapsed_seconds: float
    failed_points: int = 0
    total_points: int = 0


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
        base_x: The base parameter vector the job differentiated
            around: the line search reuses it as its own base
            (with the selected row as its direction) when the
            current parameters still match it.
        mode: The run mode the job's points ran in (`MODE_FULL` or
            `MODE_HYDRO`), shown in the page's title.
        csv_path: The path of the written matrix CSV, quoted in
            the page's caption.
        elapsed_seconds: The wall-clock duration of the job, shown
            in the completion status line.
    """

    table: pd.DataFrame
    base_x: np.ndarray
    mode: str
    csv_path: Path
    elapsed_seconds: float


class _GradientWorker(QThread):
    """
    A worker thread computing the finite-difference gradient of
    every objective function with respect to every free parameter.

    Each free parameter is perturbed up and down by 1% of its
    current range (clamped to the range's ends) and each
    perturbed point is modelled once, in a worker pool, in the
    job's run mode; the base point reuses the window's results
    when they were up to date at the job's start and is run
    otherwise.  The worker mirrors its stdout to the real one
    (so console output is unchanged) and forwards its progress
    lines to the window's status bar; on completion it emits
    ``finished_gradient`` with a `_GradientOutcome`.  Jobs cannot
    be interrupted, so the window waits for a worker before
    closing.

    Signals:
        finished_gradient: Emitted (on the main thread) with the
            `_GradientOutcome` of this job.
        progress_line: Emitted with the job's progress lines (the
            base point, then each perturbed point), for the
            window's status bar.
        progress_step: Emitted with (done, total) step counts:
            one per completed point (the base point when it was
            run, then each deduped perturbed point), for the
            window's progress bar.

    Args:
        base_x: The parameter vector to differentiate around.
        steps: The ± perturbation of every free parameter, in the
            parameter vector's key order.
        mode: The run mode (`MODE_FULL` or `MODE_HYDRO`) the
            points run in.
        csv_path: The path of the input CSV each pool worker
            loads in its initializer.
        base_results: The base point's results when they are
            already up to date on screen (None to run the base
            point too).
        parent: The parent object (the window); defaults to None.
    """

    finished_gradient = Signal(object)
    progress_line = Signal(str)
    progress_step = Signal(int, int)

    def __init__(
        self,
        base_x: np.ndarray,
        steps: list[_Perturbation],
        mode: str,
        csv_path: str,
        base_results: GuiRunResults | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._base_x: np.ndarray = base_x
        self._steps: list[_Perturbation] = steps
        self._mode: str = mode
        self._csv_path: str = csv_path
        self._base_results: GuiRunResults | None = base_results

    @staticmethod
    def _key(point: np.ndarray) -> tuple[float, ...]:
        """The identity of a parameter vector, for the results cache."""
        return tuple(float(value) for value in point)

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
        Run the job's points in a worker pool and build the
        outcome.

        The base point is run only when it was not handed in; the
        perturbed points are deduplicated against it (a point
        clamped at a range end coincides with the base) and the
        distinct points are each modelled once, in the pool.

        Each completed point is reported as a (done, total) step
        count (the base point when it was run, then each deduped
        perturbed point), for the window's progress bar.

        A failed base point fails the whole job, since the
        difference matrix needs its values; a failed non-base
        point does not: its column of the matrix is all NaN (the
        missing value) and the failure is counted, so the window
        can report it in the completion status.

        Args:
            started: The wall-clock time the job started, for the
                outcome's elapsed time.

        Returns:
            The success outcome: the difference matrix, the base
            point's results, the counts of the failed and the
            evaluated points, and the path of the written CSV; on
            a failed base point, the error outcome instead (with
            the handed-in base results when there were any).
        """
        total: int = len(self._steps) * 2

        seen: set[tuple[float, ...]] = {self._key(self._base_x)}
        points: list[np.ndarray] = []
        labels: list[str] = []
        if self._base_results is None:
            points.append(self._base_x)
            labels.append("base")
        for step in self._steps:
            for suffix, point in ((" +", step.x_plus), (" -", step.x_minus)):
                key: tuple[float, ...] = self._key(point)
                if key not in seen:
                    seen.add(key)
                    points.append(point)
                    labels.append(f"{step.key}{suffix}")

        step_total: int = len(points)
        done: int = 0

        def on_result(index: int, payload: GuiRunResults | str) -> None:
            nonlocal done
            done += 1
            self.progress_line.emit(
                f"Gradient run {done}/{total}: {labels[index]}"
            )
            self.progress_step.emit(done, step_total)

        payloads: list[GuiRunResults | str] = map_model_runs(
            points,
            self._mode,
            self._csv_path,
            on_result=on_result,
        )
        failed_points: int = 0
        results: dict[tuple[float, ...], GuiRunResults | str] = {}
        if self._base_results is not None:
            results[self._key(self._base_x)] = self._base_results
        for point, payload in zip(points, payloads):
            if isinstance(payload, str):
                if self._key(point) == self._key(self._base_x):
                    return _GradientOutcome(
                        error=payload,
                        table=None,
                        base_results=self._base_results,
                        base_x=self._base_x,
                        mode=self._mode,
                        csv_path=None,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                failed_points += 1
            results[self._key(point)] = payload
        base_payload: GuiRunResults | str = results[self._key(self._base_x)]
        if not isinstance(base_payload, GuiRunResults):
            raise ValueError(
                f"The gradient job's base point failed ({base_payload}), "
                "but the job should have stopped on that failure: "
                "the difference matrix cannot be built without the "
                "base point's values."
            )
        base_results: GuiRunResults = base_payload

        base_labels, _ = objective_vector(base_results)

        def point_vector(point: np.ndarray) -> np.ndarray:
            payload: GuiRunResults | str = results[self._key(point)]
            if isinstance(payload, GuiRunResults):
                return objective_vector(payload)[1]
            return np.full(len(base_labels), np.nan)

        f_plus: list[np.ndarray] = []
        f_minus: list[np.ndarray] = []
        for step in self._steps:
            f_plus.append(point_vector(step.x_plus))
            f_minus.append(point_vector(step.x_minus))

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
            failed_points=failed_points,
            total_points=step_total,
        )


# ==== Line search ==== #
@dataclass(frozen=True, slots=True)
class _LineSearchOutcome:
    """
    The outcome of one finished line search job.

    A successful job carries the objective's value profile (one
    row per evaluated point: the step ``t`` as index, the
    objective's value and every free parameter's value as
    columns), the best point's parameter vector and results (so
    the window can adopt them), and the path of the CSV the
    profile was written to; a failed job carries the error
    instead, with the rest None.  The objective's target, metric,
    and the job's mode ride on every outcome: the window shows
    them in the completion status line and the page's title, and
    it needs the mode to tell whether the best point's results
    are up to date.

    A failed search point does not fail the job: its entry in the
    profile is NaN (the missing value), and the counts of the
    failed and the evaluated points ride on the outcome so the
    window can report them in the completion status.  A failed
    base or direction point, by contrast, fails the whole job:
    the search needs its base point and, when the direction is
    computed, its direction points to even start.

    Attributes:
        error: The failure, formatted as ``Name: message``; None
            on success.
        profile: The value profile (the point's step ``t`` as
            index); None on failure.
        x_best: The best point's parameter vector; None on
            failure.
        best_results: The best point's results; None on failure.
        csv_path: The path of the written profile CSV; None on
            failure.
        target: The objective's target (measured series) name.
        metric: The objective's metric (the display label, e.g.
            ``KGE``).
        mode: The run mode the job's points ran in (`MODE_FULL` or
            `MODE_HYDRO`).
        elapsed_seconds: The wall-clock duration of the job.
        failed_points: The number of the job's search points that
            failed: their profile entries are NaN.  Always zero
            on a failed job.
        total_points: The number of points the job evaluated (the
            base point when it ran, the direction points when the
            direction was computed, and the search's points).
            Always zero on a failed job.
    """

    error: str | None
    profile: pd.DataFrame | None
    x_best: np.ndarray | None
    best_results: GuiRunResults | None
    csv_path: Path | None
    target: str
    metric: str
    mode: str
    elapsed_seconds: float
    failed_points: int = 0
    total_points: int = 0


class _LineSearchWorker(QThread):
    """
    One line search job on a worker thread.

    The job follows the steepest ascent of one objective function
    from the base point: when the direction was not handed in, it
    is computed by the same central differences as the gradient
    job (for the one selected row), and the objective is then
    evaluated at a fixed number of points on the ascent line, all
    inside the current parameter bounds.  The point with the
    largest finite value is the best one (the searched metrics all
    have 1 as their optimal value, so maximizing finds it).

    Every point — the base point when it was not handed in, the
    direction's perturbation points when the direction is
    computed, and the search's points — is modelled in a worker
    pool.

    Signals:
        finished_line_search: Emitted with the job's outcome.
        progress_line: Emitted with a progress line (the point
            being run), shown in the window's status line.
        progress_step: Emitted with (done, total) step counts:
            one per completed point (the base point when it was
            run, then each direction point when the direction is
            computed, then each of the search's points), for the
            window's progress bar.

    Args:
        base_x: The parameter vector to search from.
        steps: The ± perturbation of every free parameter, in the
            parameter vector's key order.
        direction: The ascent direction, or None to compute it by
            central differences.
        target: The objective's target (measured series) name.
        metric: The objective's metric (the display label, e.g.
            ``KGE``).
        mode: The run mode (`MODE_FULL` or `MODE_HYDRO`) the
            points run in.
        csv_path: The path of the input CSV each pool worker
            loads in its initializer.
        bounds: The current parameter bounds, keyed by parameter.
        base_results: The base point's results when they are
            already up to date on screen (None to run the base
            point too).
        parent: The parent object (the window); defaults to None.
    """

    finished_line_search = Signal(_LineSearchOutcome)
    progress_line = Signal(str)
    progress_step = Signal(int, int)

    def __init__(
        self,
        base_x: np.ndarray,
        steps: list[_Perturbation],
        direction: np.ndarray | None,
        target: str,
        metric: str,
        mode: str,
        csv_path: str,
        bounds: dict[str, tuple[float, float]],
        base_results: GuiRunResults | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._base_x: np.ndarray = base_x
        self._steps: list[_Perturbation] = steps
        self._direction: np.ndarray | None = direction
        self._target: str = target
        self._metric: str = metric
        self._mode: str = mode
        self._csv_path: str = csv_path
        self._bounds: dict[str, tuple[float, float]] = bounds
        self._base_results: GuiRunResults | None = base_results

    def _key(self, x: np.ndarray) -> tuple[float, ...]:
        """The deduplication key of a parameter vector (its values)."""
        return tuple(float(v) for v in x)

    def run(self) -> None:
        """Run the job and emit the outcome, capturing stdout lines."""
        # The Any annotation is the type of sys.stdout in the
        # stdlib stubs; the value is a real text stream at runtime.
        real_stdout: Any = sys.stdout
        sys.stdout = _CapturingStream(self.progress_line.emit, real_stdout)
        started: float = time.perf_counter()
        outcome: _LineSearchOutcome
        try:
            outcome = self._compute(started)
        except Exception as err:
            outcome = _LineSearchOutcome(
                error=f"{type(err).__name__}: {err}",
                profile=None,
                x_best=None,
                best_results=None,
                csv_path=None,
                target=self._target,
                metric=self._metric,
                mode=self._mode,
                elapsed_seconds=time.perf_counter() - started,
            )
        finally:
            sys.stdout = real_stdout
        self.finished_line_search.emit(outcome)

    def _point_value(self, results: GuiRunResults) -> float:
        """
        The job's objective value at one point's results, or NaN.

        Returns:
            The objective's value, or NaN when the run did not
            report it.
        """
        labels, vector = objective_vector(results)
        try:
            return float(vector[labels.index(f"{self._target}: {self._metric}")])
        except ValueError:
            return float("nan")

    def _failed_outcome(
        self, error: str, started: float
    ) -> _LineSearchOutcome:
        """
        The failed-job outcome carrying `error`.

        Args:
            error: The failure, as reported by the point's run.
            started: The wall-clock time the job started, for the
                outcome's elapsed time.
        """
        return _LineSearchOutcome(
            error=error,
            profile=None,
            x_best=None,
            best_results=None,
            csv_path=None,
            target=self._target,
            metric=self._metric,
            mode=self._mode,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _compute(self, started: float) -> _LineSearchOutcome:
        """
        Run the job's points in a worker pool and build its
        success outcome.

        The base point is run only when it was not handed in.
        When the direction was not handed in either, the pool
        first runs the direction's points — every free parameter
        perturbed up and down by 1% of its range (clamped to the
        range's ends), deduplicated against the base point like
        the gradient job — and the direction is then the selected
        objective's row of their difference matrix, normalized to
        unit length and expanded to the parameter vector's space
        (a pinned parameter's component stays zero).

        The search then evaluates the objective at
        `_LINE_SEARCH_POINTS` further pool points on the ascent
        line ``x + t * direction``, ``t`` equally spaced across
        the interval that keeps every parameter inside its
        bounds, and the point with the largest finite value is
        the best.

        Each completed point is reported as a (done, total) step
        count (the base point when it was run, then each
        direction point when the direction is computed, then each
        of the search's points), for the window's progress bar.

        A failed base or direction point — including a failed
        pool initializer, which fails every point its worker
        receives — fails the whole job with the first failure's
        message, in the points' input order.  A failed search
        point does not fail the job: its entry in the profile is
        NaN (the missing value) and the failure is counted, so
        the window can report it in the completion status; the
        search for the best point skips NaN entries, and a job
        whose search points all failed raises the error instead.

        Args:
            started: The wall-clock time the job started, for the
                outcome's elapsed time.

        Returns:
            The success outcome: the value profile, the best
            point's vector and results, the counts of the failed
            and the evaluated points, and the path of the written
            CSV; on a failed base or direction point, the error
            outcome instead.

        Raises:
            ValueError:
                Raised when the selected objective has no finite
                value at the base point, when its differences
                carry no direction, or when no point on the
                ascent line lies inside the current bounds (or
                reports a value).
        """
        total: int = len(self._steps) * 2

        # The first pool phase: the base point when it was not
        # handed in, and — when the direction must be computed —
        # the direction's deduplicated perturbation points, the
        # same set the gradient job would run.
        phase1_points: list[np.ndarray] = []
        phase1_labels: list[str] = []
        if self._base_results is None:
            phase1_points.append(self._base_x)
            phase1_labels.append("base")
        if self._direction is None:
            seen: set[tuple[float, ...]] = {self._key(self._base_x)}
            for step in self._steps:
                for suffix, point in (
                    (" +", step.x_plus),
                    (" -", step.x_minus),
                ):
                    key: tuple[float, ...] = self._key(point)
                    if key not in seen:
                        seen.add(key)
                        phase1_points.append(point)
                        phase1_labels.append(f"{step.key}{suffix}")

        step_total: int = len(phase1_points) + _LINE_SEARCH_POINTS
        done: int = 0

        def on_phase1_result(
            index: int, payload: GuiRunResults | str
        ) -> None:
            nonlocal done
            done += 1
            self.progress_line.emit(
                f"Line search run {done}/{total}: {phase1_labels[index]}"
            )
            self.progress_step.emit(done, step_total)

        phase1_payloads: list[GuiRunResults | str] = []
        if phase1_points:
            phase1_payloads = map_model_runs(
                phase1_points,
                self._mode,
                self._csv_path,
                on_result=on_phase1_result,
            )
        results: dict[tuple[float, ...], GuiRunResults] = {}
        if self._base_results is not None:
            results[self._key(self._base_x)] = self._base_results
        for point, payload in zip(phase1_points, phase1_payloads):
            if isinstance(payload, str):
                return self._failed_outcome(payload, started)
            results[self._key(point)] = payload
        base_results: GuiRunResults = results[self._key(self._base_x)]

        if self._direction is not None:
            direction: np.ndarray = self._direction
        else:
            direction = self._difference_direction(base_results, results)

        interval: tuple[float, float] | None = feasible_t_interval(
            self._base_x, direction, self._steps, self._bounds
        )
        if interval is None:
            raise ValueError(
                "The line search found no feasible step: every point "
                "on the ascent line lies outside the current "
                "parameter bounds."
            )
        t_lo, t_hi = interval
        t_points: np.ndarray = np.linspace(t_lo, t_hi, _LINE_SEARCH_POINTS)
        search_points: list[np.ndarray] = [
            self._base_x + t * direction for t in t_points
        ]
        search_labels: list[str] = [f"t = {t:.4f}" for t in t_points]

        def on_search_result(
            index: int, payload: GuiRunResults | str
        ) -> None:
            nonlocal done
            done += 1
            self.progress_line.emit(
                f"Line search run {done}/{total}: {search_labels[index]}"
            )
            self.progress_step.emit(done, step_total)

        search_payloads: list[GuiRunResults | str] = map_model_runs(
            search_points,
            self._mode,
            self._csv_path,
            on_result=on_search_result,
        )

        failed_points: int = 0
        values: list[float] = []
        point_results: list[GuiRunResults | str] = []
        for payload in search_payloads:
            point_results.append(payload)
            if isinstance(payload, str):
                failed_points += 1
                values.append(float("nan"))
            else:
                values.append(self._point_value(payload))
        values_array: np.ndarray = np.array(values, dtype=np.float64)
        if not np.isfinite(values_array).any():
            raise ValueError(
                f"Every point on the ascent line failed to report a "
                f"value for {self._target}: {self._metric}."
            )
        best_index: int = int(np.nanargmax(values_array))
        x_best: np.ndarray = self._base_x + t_points[best_index] * direction
        best_payload: GuiRunResults | str = point_results[best_index]
        if not isinstance(best_payload, GuiRunResults):
            raise ValueError(
                f"The best search point's run failed ({best_payload}), "
                "but the search should have skipped it: a NaN value "
                "can never be the largest finite value."
            )
        best_results: GuiRunResults = best_payload

        columns: dict[str, np.ndarray] = {"value": values_array}
        for step in self._steps:
            position: int = list(self._bounds).index(step.key)
            columns[step.key] = self._base_x[position] + t_points * direction[
                position
            ]
        profile: pd.DataFrame = pd.DataFrame(columns, index=t_points)
        csv_path: Path = write_line_search_csv(profile)
        return _LineSearchOutcome(
            error=None,
            profile=profile,
            x_best=x_best,
            best_results=best_results,
            csv_path=csv_path,
            target=self._target,
            metric=self._metric,
            mode=self._mode,
            elapsed_seconds=time.perf_counter() - started,
            failed_points=failed_points,
            total_points=step_total,
        )

    def _difference_direction(
        self,
        base_results: GuiRunResults,
        results: dict[tuple[float, ...], GuiRunResults],
    ) -> np.ndarray:
        """
        The normalized steepest-ascent direction of the job's
        objective, from its direction points' central
        differences.

        The direction's points — every free parameter perturbed
        up and down by 1% of its range (clamped to the range's
        ends), deduplicated against the base point, like the
        gradient job — were already run in the job's first pool
        phase; their results, together with the base point's,
        are `results`.  This method reads them only, so it runs
        no points and reports no progress of its own.  The
        direction is the selected objective's row of the
        difference matrix, normalized to unit length and expanded
        to the parameter vector's space (a pinned parameter's
        component stays zero).

        Args:
            base_results: The base point's results (its objective
                vector carries the row's label).
            results: The first phase's results, keyed by each
                point's value tuple (the base point's included).

        Returns:
            The unit-length direction vector, in the parameter
            vector's order.

        Raises:
            ValueError:
                Raised when the base point did not report the
                selected objective, when its differences are not
                all finite, or when they are all zero (no
                direction).
        """
        base_labels, _ = objective_vector(base_results)
        try:
            row_index: int = base_labels.index(
                f"{self._target}: {self._metric}"
            )
        except ValueError:
            raise ValueError(
                f"The base point did not report the selected objective "
                f"{self._target}: {self._metric}."
            ) from None
        f_plus: list[np.ndarray] = []
        f_minus: list[np.ndarray] = []
        for step in self._steps:
            _, up_vector = objective_vector(results[self._key(step.x_plus)])
            _, down_vector = objective_vector(
                results[self._key(step.x_minus)]
            )
            f_plus.append(up_vector)
            f_minus.append(down_vector)

        matrix: np.ndarray = gradient_matrix(f_plus, f_minus, self._steps)
        row: np.ndarray = matrix[row_index]
        if not np.isfinite(row).all():
            raise ValueError(
                f"The selected objective {self._target}: {self._metric} "
                "has a missing difference at the base point, so its "
                "direction cannot be computed."
            )
        norm: float = float(np.linalg.norm(row))
        if norm == 0.0:
            raise ValueError(
                f"The selected objective {self._target}: {self._metric} "
                "is flat at the base point: its differences are all "
                "zero, so the line search has no direction."
            )
        return expand_direction(row / norm, self._steps, self._bounds)


@dataclass(frozen=True, slots=True)
class _LineSearchSnapshot:
    """
    The last finished line search job's results, kept for the
    figure panel.

    The profile is re-drawn as the figure panel's extra page every
    time the figures are rebuilt, so it stays on screen across
    runs, mode changes, and plot-selection changes until the next
    successful line search replaces it.

    Attributes:
        profile: The value profile (the point's step ``t`` as
            index, the objective's value and every free
            parameter's value as columns).
        target: The objective's target (measured series) name,
            shown in the page's title.
        metric: The objective's metric (the display label, e.g.
            ``KGE``), shown in the page's title.
        mode: The run mode the job's points ran in (`MODE_FULL` or
            `MODE_HYDRO`), shown in the page's title.
        csv_path: The path of the written profile CSV, quoted in
            the page's caption.
        elapsed_seconds: The wall-clock duration of the job, shown
            in the completion status line.
    """

    profile: pd.DataFrame
    target: str
    metric: str
    mode: str
    csv_path: Path
    elapsed_seconds: float


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


class _StatusLabel(QLabel):
    """
    A status line that cannot force the window beyond the screen.

    A plain label reports a minimum size hint equal to its full
    text, so a long message (a run failure, for instance) would
    push the window's minimum width past a small screen, and the
    window could never be squeezed back below it.  This label
    reports a small constant minimum hint instead: the header row
    then squeezes the label to that width (clipping the text)
    rather than growing, and the full message stays in the
    label's tooltip.  The size hint still follows the text, so the
    label shows the whole message whenever the window has room
    for it.
    """

    def minimumSizeHint(self) -> QSize:
        """The minimum: a small fixed width and the hint's height."""
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), _STATUS_MIN_WIDTH), hint.height())


class _SpinnerLabel(QLabel):
    """
    The header's running-job indicator: a small square label that
    cycles through its frames while a job (a run, a gradient
    computation, or a line search) is in flight.

    The label is hidden while no job runs and has a fixed square
    footprint (see `_SPINNER_SIZE`), so showing or hiding it never
    changes the header's minimum width and never resizes the
    window.  While spinning, a timer advances the displayed frame
    on each tick (see `_SPINNER_INTERVAL_MS`).

    Args:
        parent: The parent widget; defaults to None.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_index: int = 0
        self.setFixedSize(_SPINNER_SIZE, _SPINNER_SIZE)
        font: QFont = QFont(self.font())
        font.setBold(True)
        self.setFont(font)
        self._timer: QTimer = QTimer(self)
        self._timer.setInterval(_SPINNER_INTERVAL_MS)
        self._timer.timeout.connect(self._advance_frame)
        self.hide()

    def set_spinning(self, enabled: bool) -> None:
        """
        Show or hide the indicator.

        Enabling shows the label at its current frame and starts
        the frame-advance timer; disabling stops the timer, hides
        the label, and clears the frame.

        Args:
            enabled: Whether a job is in flight.
        """
        if enabled:
            self._show_frame()
            self._timer.start()
        else:
            self._timer.stop()
            self.setText("")
            self.hide()

    def _show_frame(self) -> None:
        """Set the current frame and make the label visible."""
        self.setText(_SPINNER_FRAMES[self._frame_index])
        self.show()

    def _advance_frame(self) -> None:
        """Advance to the next frame of the cycle."""
        self._frame_index = (self._frame_index + 1) % len(_SPINNER_FRAMES)
        self._show_frame()


# ==== Main window ==== #
class MainWindow(QMainWindow):
    """
    The main window of the East River lateral-model calibration GUI.

    The window stacks two non-collapsible sections: the figure
    panel (the model's figures, one visible at a time: the three
    full-mode figures, or the streamflow and zone-storage figures
    in hydrology-only mode, plus the gradient and line search
    pages when those jobs have finished) and, below it, a row
    with the fixed-size per-run objective-function table on the
    left and the parameter panel on the right.  The header row
    carries the title, a *Hydrology only* checkbox next to it (off
    by default), a status line, a *Save run history* checkbox (off
    by default), an *Auto-run* checkbox, and the *Run model*,
    *Gradients*, *Line search*, *Reset bounds*, and *Save run*
    buttons.  A thin strip along the window's bottom edge shows an
    indeterminate progress bar and the elapsed time while a job is
    in flight; the job indicators are kept out of the header so
    the window's width never depends on a job's state or the
    length of a status message.

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
    the figures reduce to the streamflow figure and a figure of
    each zone's water storage, and the objective table carries
    only the streamflow column.  A mode change stales the results
    on screen just like a parameter change does.  A run that
    fails (for instance with ``WaterVolumeError``) is reported in
    the status bar, and a water-volume failure additionally as a
    modal dialog, and the last valid figures stay on screen.
    *Run model*, *Gradients*, and *Line search* are disabled
    while a job is in flight; *Run model* is also disabled while
    the results on screen already correspond to the current
    parameters.

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
        self._worker: (
            _RunWorker | _GradientWorker | _LineSearchWorker | None
        ) = None
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
        # panel draws it as an extra page and the line search
        # reuses its matrix as the search direction.
        self._gradient_state: _GradientSnapshot | None = None
        # The last finished line search job's profile and metadata
        # (None while no line search has finished); the figure
        # panel draws it as an extra page.
        self._line_search_state: _LineSearchSnapshot | None = None

        self.setWindowTitle(_TITLE)
        self.resize(*_INITIAL_SIZE)

        self._build_ui()
        self._build_actions()
        self._restore_or_start()

    # ==== Building ==== #
    def _build_ui(self) -> None:
        """Assemble the header row, the figure, the lower row, and the progress strip."""
        header: QWidget = self._build_header()

        self._figure_panel: FigurePanel = self._build_figure_panel()
        self._build_objective_panel()
        param_panel: ParamPanel = self._build_param_panel()

        # The main splitter stacks the figure over the lower row,
        # which is a plain row: the fixed-size objective panel on
        # the left and the parameter panel (stretching) on the
        # right.  There is no inner splitter: the objective panel
        # is fixed in size, so a seam there could only move, never
        # resize what is on either side of it.
        self._splitter: QSplitter = QSplitter(Qt.Orientation.Vertical)
        row: QWidget = QWidget(self)
        row_layout: QHBoxLayout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        # The fixed-size panel is top-aligned: its contents start at
        # the row's top and it never grows to fill the row's height.
        row_layout.addWidget(
            self._objective_panel,
            0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
        )
        row_layout.addWidget(param_panel, 1)
        self._splitter.addWidget(self._figure_panel)
        self._splitter.addWidget(row)
        for section in range(self._splitter.count()):
            self._splitter.setCollapsible(section, False)
        # Without an explicit distribution Qt lays the sections out
        # from their size hints, which squashes the figure (the
        # canvases report a small hint) into a sliver while the lower
        # row (large hints) takes the rest.
        self._fit_objective_panel()
        self._splitter.setSizes(list(_INITIAL_SPLIT_SIZES))
        self._figure_panel.setMinimumHeight(_FIGURE_MIN_HEIGHT)

        self._build_progress_strip()
        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(header)
        layout.addWidget(self._splitter, 1)
        layout.addWidget(self._progress_strip)
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
        """
        Build the header row: title, running indicator, status, and
        buttons.

        The row carries the fixed-size running indicator (a
        `_SpinnerLabel`) next to the title: its footprint does not
        change with the job state, so the window's minimum width is
        the same whether or not a job is in flight (the remaining
        job indicators live in the bottom strip).  The status label
        is the row's only compressible item: it reports a small
        minimum (see `_StatusLabel`) and stretches between the mode
        checkbox and the job checkboxes, so a narrow window clips
        its text (the tooltip keeps the full message) instead of
        pushing the window past the screen.
        """
        title: QLabel = QLabel(_TITLE)
        font: QFont = QFont(title.font())
        font.setBold(True)
        font.setPointSize(max(9, font.pointSize() + 2))
        title.setFont(font)

        # The running-job indicator: hidden at rest, its fixed size
        # keeps the row's footprint constant either way.
        self._job_spinner: _SpinnerLabel = _SpinnerLabel()

        # Off by default: the models run in full mode.
        self._mode_checkbox: QCheckBox = QCheckBox("Hydrology only")
        self._mode_checkbox.setToolTip(
            "Run only the hydrology model, scored against the "
            "measured streamflow.  When checked, the reactive-"
            "transport parameter groups and the species / zone / "
            "mineral figure checkboxes are hidden, the figures "
            "reduce to the streamflow figure and a figure of each "
            "zone's water storage, and the objective table carries "
            "only the streamflow column."
        )
        self._mode_checkbox.toggled.connect(self._on_mode_changed)

        self._status_label: _StatusLabel = _StatusLabel("Preparing…")
        self._status_label.setStyleSheet("color: gray;")
        self._status_label.setMaximumWidth(_STATUS_MAX_WIDTH)

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

        self._line_search_button: QPushButton = QPushButton("Line search")
        self._line_search_button.clicked.connect(self._on_line_search_clicked)
        self._line_search_button.setToolTip(
            "Follow the steepest ascent of one objective function "
            "from the current parameters: pick an objective from "
            "the menu, and the job evaluates it at 25 points on "
            "the ascent line, all inside the current parameter "
            "bounds, and adopts the point with the best value "
            "(the searchable metrics have 1 as their optimal "
            "value, so the search maximizes the metric).  When a "
            "gradient job has already run at these parameters its "
            "matrix supplies the direction; otherwise the "
            "direction is computed by the same central "
            "differences.  The value profile is written to a CSV "
            "in the results directory and drawn as an extra page "
            "of the figure panel."
        )

        self._reset_button: QPushButton = QPushButton("Reset bounds")
        self._reset_button.clicked.connect(self._on_reset_bounds)

        self._save_button: QPushButton = QPushButton("Save run")
        self._save_button.clicked.connect(self._on_save_run)
        self._save_button.setEnabled(False)

        layout: QHBoxLayout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(title)
        layout.addWidget(self._job_spinner)
        layout.addWidget(self._mode_checkbox)
        layout.addStretch(1)
        # The status label stretches with the middle stretch (both
        # factor 1) up to its maximum width; beyond that the middle
        # stretch absorbs the rest, so the label can grow but never
        # forces the row wider than the other items.
        layout.addWidget(self._status_label, 1)
        layout.addWidget(self._save_history_checkbox)
        layout.addWidget(self._auto_run_checkbox)
        layout.addWidget(self._run_button)
        layout.addWidget(self._gradients_button)
        layout.addWidget(self._line_search_button)
        layout.addWidget(self._reset_button)
        layout.addWidget(self._save_button)
        header: QWidget = QWidget(self)
        header.setLayout(layout)
        return header

    def _build_progress_strip(self) -> None:
        """
        Build the thin strip along the window's bottom edge.

        The strip carries the progress bar and the elapsed-time
        label; it is hidden except while a job (a run, a gradient
        computation, or a line search) is in flight.  The bar is
        indeterminate at rest and becomes determinate while a job
        reports its completed steps (`progress_step`); the job's
        finish handlers restore the indeterminate rest state.
        Keeping the job indicators out of the header row means the
        header's width — and with it the window's minimum width —
        is the same whether or not a job is running, so switching
        between run modes can never push the window past the
        screen.
        """
        self._progress_bar: QProgressBar = QProgressBar()
        # Range (0, 0): the indeterminate (animated) rest state; a
        # job's progress_step counts make the bar determinate (see
        # `_on_progress_step`) and its finish handlers restore
        # this state.
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._elapsed_label: QLabel = QLabel("")
        self._progress_strip: QWidget = QWidget(self)
        strip_layout: QHBoxLayout = QHBoxLayout(self._progress_strip)
        strip_layout.setContentsMargins(8, 4, 8, 4)
        strip_layout.setSpacing(8)
        strip_layout.addWidget(self._progress_bar, 1)
        strip_layout.addWidget(self._elapsed_label)
        self._progress_strip.hide()

    def _on_progress_step(self, done: int, total: int) -> None:
        """
        Update the progress bar from a job's step count.

        When the job reports a non-zero total step count, the bar
        becomes determinate for it (its range matches the total and
        its percentage is shown) and advances to the number of
        completed steps; a job that reports no total leaves the bar
        indeterminate.

        Args:
            done: The number of steps the job has completed.
            total: The job's total step count (0 if it reports
                none).
        """
        if total > 0 and self._progress_bar.maximum() != total:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setTextVisible(True)
        self._progress_bar.setValue(done)

    def _build_figure_panel(self) -> FigurePanel:
        """Build the figure panel (its header carries the arrow buttons)."""
        return FigurePanel(self)

    def _build_objective_panel(self) -> None:
        """
        Build the fixed-size objective panel, left of the parameter
        panel.

        The panel holds the objective table's heading and the table
        itself and nothing else: the metric note that used to sit
        under the table now lives in the heading's tooltip, so the
        panel stays as small as the table needs to be, and the
        window's width is driven by the parameter panel instead of
        the table.
        """
        self._objective_model: _ObjectiveTableModel = _ObjectiveTableModel(self)
        self._objective_table: QTableView = QTableView(self)
        self._objective_table.setModel(self._objective_model)
        self._objective_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._objective_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._objective_table.setShowGrid(False)
        # The panel is sized exactly to the table's contents (see
        # `_fit_objective_panel`), so the view never needs to
        # scroll: the bar policies stay off so no bar space is ever
        # reserved.
        self._objective_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._objective_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # The row headers (the metric labels) identify what each row
        # measures, so they stay visible.
        self._objective_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._objective_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        self._objective_heading: QLabel = QLabel("<b>Objective functions</b>")
        self._objective_heading.setToolTip(METRIC_NOTES)

        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(*_OBJECTIVE_PANEL_MARGINS)
        layout.setSpacing(_OBJECTIVE_PANEL_SPACING)
        layout.addWidget(self._objective_heading)
        layout.addWidget(self._objective_table)
        self._objective_panel: QWidget = QWidget(self)
        self._objective_panel.setLayout(layout)

    def _fit_objective_panel(self) -> None:
        """
        Fix the objective panel's size to fit its heading and table.

        The panel is the only lower-row widget without stretch, so
        its size drives the row's width: sizing it exactly to its
        contents (plus the panel's margins) keeps the window's
        width from growing with the table, and leaves the rest of
        the row to the stretching parameter panel.  The size is the
        table's own measurements (per-column and per-row section
        size hints, both headers' size hints, and the frame width)
        run through `_objective_panel_size`, so the scrollbars
        (held off) never have work to do.
        """
        width, height = _objective_panel_size(
            self._objective_heading.sizeHint(),
            [
                self._objective_table.horizontalHeader().sectionSizeHint(c)
                for c in range(self._objective_model.columnCount())
            ],
            [
                self._objective_table.verticalHeader().sectionSizeHint(r)
                for r in range(self._objective_model.rowCount())
            ],
            self._objective_table.verticalHeader().sizeHint().width(),
            self._objective_table.horizontalHeader().sizeHint().height(),
            self._objective_table.frameWidth(),
            _OBJECTIVE_PANEL_MARGINS,
            _OBJECTIVE_PANEL_SPACING,
        )
        self._objective_panel.setFixedSize(width, height)

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
        self._progress_strip.show()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setValue(0)
        self._job_spinner.set_spinning(True)
        self._run_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._gradients_button.setEnabled(False)
        self._line_search_button.setEnabled(False)
        self._status("Running the model…")

        worker: _RunWorker = _RunWorker(
            self._run_generation, x, self._input_data, mode, self
        )
        worker.progress_line.connect(self._on_progress_line)
        worker.progress_step.connect(self._on_progress_step)
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
        self._progress_strip.hide()
        self._job_spinner.set_spinning(False)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setValue(0)

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
            if isinstance(outcome.exception, WaterVolumeError):
                # The water-volume failure is the one the user most
                # needs to act on, so it gets a modal dialog as well
                # as the status line.
                QMessageBox.critical(
                    self, "Water volume error", str(outcome.exception)
                )
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
        # No run is in flight any more, so the gradient and the
        # line search jobs can start again (a stale restart below
        # re-disables both via `_start_run`).
        self._gradients_button.setEnabled(True)
        self._line_search_button.setEnabled(True)
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
        self._progress_strip.show()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setValue(0)
        self._job_spinner.set_spinning(True)
        self._run_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._gradients_button.setEnabled(False)
        self._line_search_button.setEnabled(False)
        self._status("Computing gradients…")

        worker: _GradientWorker = _GradientWorker(
            x,
            steps,
            self._current_mode(),
            self._csv_path or str(DEFAULT_CSV_PATH),
            base_results,
            self,
        )
        worker.progress_line.connect(self._on_gradient_progress_line)
        worker.progress_step.connect(self._on_progress_step)
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
        self._progress_strip.hide()
        self._run_started_at = None
        self._job_spinner.set_spinning(False)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setValue(0)

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
                base_x=outcome.base_x,
                mode=outcome.mode,
                csv_path=csv_path,
                elapsed_seconds=outcome.elapsed_seconds,
            )
            self._refresh_from_results()
            message: str = (
                f"Gradients finished in {outcome.elapsed_seconds:.1f} s — "
                f"saved to {csv_path}."
            )
            if outcome.failed_points > 0:
                message += (
                    f" {outcome.failed_points} of {outcome.total_points} "
                    "points failed; their values are shown as missing."
                )
            self._status(message)

        # No job is in flight by the time this handler runs (the
        # worker's finished signal clears the reference after this
        # one, like in `_on_finished_run`): the run and the save
        # buttons mirror the results' staleness and the gradients
        # and line search buttons always come back up.
        up_to_date: bool = self._results_are_up_to_date()
        self._save_button.setEnabled(up_to_date)
        self._run_button.setEnabled(not up_to_date)
        self._gradients_button.setEnabled(True)
        self._line_search_button.setEnabled(True)

    # ==== Line search ==== #
    def _on_line_search_clicked(self) -> None:
        """
        Show the line search's menu from the *Line search* button.

        The menu lists every objective function for which the
        current results report a finite value, among the metrics
        the search can optimize (the ones with 1 as their optimal
        value: KGE, LogKGE, NSE, and LogNSE); picking one starts a
        line search of that objective.  A job already in flight
        takes precedence: the click is ignored, since a job cannot
        be interrupted.
        """
        if self._worker is not None and self._worker.isRunning():
            return
        table: pd.DataFrame = self._objective_model._table
        if table.empty:
            self._status(
                "No objective values to search yet: run the model first."
            )
            return
        candidates: list[tuple[str, str]] = [
            (str(target), str(metric))
            for target in table.columns
            for metric in table.index
            if metric in _LINE_SEARCH_METRICS
            and np.isfinite(table.loc[metric, target])
        ]
        if not candidates:
            self._status(
                "No search objective is available: run the model first."
            )
            return
        menu: QMenu = QMenu(self)
        for target, metric in candidates:
            action: QAction = menu.addAction(f"{target}: {metric}")
            action.triggered.connect(
                partial(self._on_line_search_selected, target, metric)
            )
        menu.exec(
            self._line_search_button.mapToGlobal(
                self._line_search_button.rect().bottomLeft()
            )
        )

    def _on_line_search_selected(
        self, target: str, metric: str, _checked: bool
    ) -> None:
        """Start a line search of the objective picked from the menu."""
        self._start_line_search(target, metric)

    def _start_line_search(self, target: str, metric: str) -> None:
        """
        Start a line search of one objective function.

        The search follows the steepest ascent of the objective
        from the current parameters: when a gradient job has
        already finished at these parameters (in the same mode,
        with the same free parameters), its matrix supplies the
        direction for free; otherwise the direction is computed by
        the same central differences as the gradient job, for the
        one selected row.  The objective is then evaluated at 25
        points on the ascent line, all inside the current
        parameter bounds, and the point with the best (largest)
        value is adopted.

        The run indicators show the job's progress like a model
        run's; a job cannot be interrupted.

        A job already in flight (a model run or a gradient job)
        takes precedence: the search is refused with a status
        message, since a job cannot be interrupted.
        """
        if self._worker is not None and self._worker.isRunning():
            self._status(
                "A job is already in flight: it cannot be interrupted, "
                "so the line search was not started."
            )
            return
        x: np.ndarray = self._current_x()
        bounds: dict[str, tuple[float, float]] = (
            self._param_panel.current_bounds()
        )
        steps: list[_Perturbation] = free_parameter_steps(x, bounds)
        if not steps:
            self._status("No free parameters to search.")
            return
        direction: np.ndarray | None = None
        snapshot: _GradientSnapshot | None = self._gradient_state
        if (
            snapshot is not None
            and snapshot.mode == self._current_mode()
            and np.array_equal(snapshot.base_x, x)
            and len(snapshot.table.columns) == len(steps)
        ):
            try:
                row: np.ndarray = snapshot.table.loc[
                    f"{target}: {metric}"
                ].to_numpy(dtype=np.float64)
                direction = expand_direction(row, steps, bounds)
            except KeyError:
                direction = None
        base_results: GuiRunResults | None = (
            self._results
            if self._results is not None and self._results_are_up_to_date()
            else None
        )

        self._run_started_at = time.perf_counter()
        self._progress_strip.show()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setValue(0)
        self._job_spinner.set_spinning(True)
        self._run_button.setEnabled(False)
        self._save_button.setEnabled(False)
        self._gradients_button.setEnabled(False)
        self._line_search_button.setEnabled(False)
        self._status(f"Line search in progress: {target}: {metric} …")
        self._worker = _LineSearchWorker(
            x,
            steps,
            direction,
            target,
            metric,
            self._current_mode(),
            self._csv_path or str(DEFAULT_CSV_PATH),
            bounds,
            base_results,
            self,
        )
        worker: _LineSearchWorker = self._worker
        worker.progress_line.connect(self._on_line_search_progress_line)
        worker.progress_step.connect(self._on_progress_step)
        worker.finished_line_search.connect(self._on_finished_line_search)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(worker.deleteLater)
        self._elapsed_timer.start()
        worker.start()

    def _on_line_search_progress_line(self, line: str) -> None:
        """
        Show the line search's latest progress line in the status
        bar.
        """
        if line:
            self._status(f"Line search in progress — {line}")

    def _on_finished_line_search(self, payload: object) -> None:
        """
        Apply a finished line search job's outcome to the window.

        A failed job is reported in the status bar and leaves the
        figures on screen unchanged.  A finished job's profile is
        kept as the figure panel's extra page, its best point's
        parameters and results are adopted into the window (the
        panel is restored to the best point's values and the
        figures and table refresh), and its completion is reported
        in the status bar with the best step, the best value, and
        the path of the written CSV.
        """
        self._elapsed_timer.stop()
        self._progress_strip.hide()
        self._run_started_at = None
        self._job_spinner.set_spinning(False)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setValue(0)
        self._gradients_button.setEnabled(True)
        self._line_search_button.setEnabled(True)

        if not isinstance(payload, _LineSearchOutcome):
            return
        outcome: _LineSearchOutcome = payload
        if (
            outcome.error is not None
            or outcome.profile is None
            or outcome.x_best is None
            or outcome.best_results is None
            or outcome.csv_path is None
        ):
            self._status(f"Line search failed: {outcome.error}")
            # No job is in flight by the time this handler runs (the
            # worker's finished signal clears the reference after
            # this one, like in `_on_finished_gradient`): the run
            # and the save buttons mirror the results' staleness.
            up_to_date: bool = self._results_are_up_to_date()
            self._save_button.setEnabled(up_to_date)
            self._run_button.setEnabled(not up_to_date)
            return

        # The best point is the profile's largest finite value.
        values: np.ndarray = outcome.profile["value"].to_numpy(
            dtype=np.float64
        )
        best_index: int = int(np.nanargmax(values))
        best_t: float = float(outcome.profile.index[best_index])
        best_value: float = float(values[best_index])

        bounds: dict[str, tuple[float, float]] = (
            self._param_panel.current_bounds()
        )
        best_values: dict[str, float] = {
            key: float(value)
            for key, value in zip(bounds, outcome.x_best, strict=True)
        }
        self._param_panel.restore_state(best_values, bounds)
        self._results = outcome.best_results
        self._last_run_x = outcome.x_best
        self._last_run_mode = outcome.mode
        self._line_search_state = _LineSearchSnapshot(
            profile=outcome.profile,
            target=outcome.target,
            metric=outcome.metric,
            mode=outcome.mode,
            csv_path=outcome.csv_path,
            elapsed_seconds=outcome.elapsed_seconds,
        )
        self._refresh_from_results()
        message: str = (
            f"Line search finished in {outcome.elapsed_seconds:.1f} s: "
            f"best t = {best_t:.4f} ({outcome.target}: "
            f"{outcome.metric} = {best_value:.3f}); the best point's "
            f"parameters are in the panel and the profile was saved "
            f"to {outcome.csv_path}."
        )
        if outcome.failed_points > 0:
            message += (
                f" {outcome.failed_points} of {outcome.total_points} "
                "points failed; their values are shown as missing."
            )
        self._status(message)
        # The save button lights up only when the best point is
        # where the panel now sits and the job ran in the current
        # mode; the run button mirrors the same staleness.
        up_to_date = self._results_are_up_to_date()
        self._save_button.setEnabled(up_to_date)
        self._run_button.setEnabled(not up_to_date)

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
        Save the current run: the existing per-run model output,
        the figures as plotted, and a ``gui_state.json`` capturing
        the session state.

        The run is saved only when the results on screen correspond
        to the current parameters and mode (a run still catching up
        blocks the save, since its output would be stale).  A
        full-mode run saves the ``hydro`` and ``rt`` bundle pair
        and a hydrology-only run the flat hydrology bundle; the
        state file records the run mode.  Every figure currently
        loaded in the figure panel (the mode's own figures, and the
        gradient and line search pages when those jobs have
        finished) is saved as a PNG into a ``figures`` subfolder of
        the run directory.  The new run directory becomes the
        first entry in the panel's *History* group (the newest
        save), so it can be restored from the dropdown without a
        re-run.
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
                int(size) for size in self._splitter.sizes()
            ],
            save_history=self._save_history_checkbox.isChecked(),
            mode=self._current_mode(),
        )
        write_gui_state(run_dir, state)
        figure_paths: list[Path] = self._figure_panel.save_pages(
            run_dir / "figures"
        )
        self._add_saved_history_item(run_dir)
        self._status(
            f"Saved run to {run_dir}; {len(figure_paths)} figure "
            f"page(s) saved to {run_dir / 'figures'}."
        )

    # ==== Refresh ==== #
    def _refresh_from_results(self) -> None:
        """
        Rebuild the figures and the objective table from the last
        results.

        Every figure honors the current plot selection, so a
        selection change refreshes them without a re-run.  A
        hydrology-only run's results carry no reactive-transport
        simulation, so the streamflow figure and a figure of each
        zone's water storage are shown and the objective table
        carries only the streamflow column.  A previous run
        selected in the panel's *History* group (with its overlay
        checkbox on) is drawn faintly on every figure, behind the
        current run's series.  When a gradient job has finished,
        its difference matrix is drawn as an extra page after the
        mode's own figures, and when a line search has finished,
        its value profile rides along as a further page.
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
            # A hydrology-only run shows the streamflow figure and
            # each zone's water storage, with no reactive-transport
            # column in the table.
            time_series: Figure = make_time_series_figure(
                hydro_sim,
                None,
                self._input_data.q_meas,
                self._input_data.conc_df,
                selection.time_series,
                overlay_hydro_sim=overlay_hydro,
                overlay_rt_sim=overlay_rt,
            )
            storage: Figure = make_zone_figure(
                None,
                hydro_sim,
                [],
                selection.zones,
                overlay_rt_sim=overlay_rt,
                overlay_hydro_sim=overlay_hydro,
            )
            figures: tuple[tuple[str, Figure], ...] = (
                ("Streamflow", time_series),
                ("Zone storage", storage),
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

        # The last finished line search's profile rides along as an
        # extra page, like the gradient matrix.
        if self._line_search_state is not None:
            search: _LineSearchSnapshot = self._line_search_state
            note = (
                f"Steepest ascent of {search.target}: {search.metric}, "
                f"{_LINE_SEARCH_POINTS} points within the current "
                f"bounds; saved to {search.csv_path}."
            )
            mode_label = "hydro" if search.mode == MODE_HYDRO else "full"
            search_fig: Figure = make_line_search_figure(
                search.profile, search.target, search.metric, note
            )
            figures = figures + (
                (f"Line search ({mode_label})", search_fig),
            )

        self._figure_panel.set_figures(figures)
        self._objective_model.set_table(table)
        # The objective panel is fixed in size; re-fit it after the
        # table's contents (and thus its size hint) changed.
        self._fit_objective_panel()

    # ==== Status ==== #
    def _status(self, text: str) -> None:
        """
        Set the status line of the header row.

        The label clips text longer than its maximum width, so the
        full text is mirrored into the label's tooltip.
        """
        self._status_label.setText(text)
        self._status_label.setToolTip(text)

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
