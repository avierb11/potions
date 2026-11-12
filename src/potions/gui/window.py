"""
The main window of the East River lateral-model calibration GUI.

The window stacks two sections: the figure panel (the model's
figures, one visible at a time, flipped by the panel's arrow
buttons or the main window's Left / Right keys: the hydrology,
storage, and partitioning figures, joined by the river
concentrations, river loads, concentration vs. discharge, zone
concentrations, and reaction rates figures in full mode, plus the
gradient job's matrix and a line search's value profile as extra
pages when those jobs have finished) and, below it, a row holding the fixed-size per-run
objective-function table on the left and the parameter panel on
the right.  Home+Up toggles the window's maximize.

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

The header's *Calibrate* button (available only in hydrology-only
mode) optimizes the free hydrology parameters with differential
evolution against the checked objective terms: a small dialog asks
for the generation count, population size, and whether to polish,
and the best run's parameters are adopted into the panel, its
figures refreshed, and the run saved.  A pinned parameter (a cell
whose range has no width) keeps its current value.

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
window opens without a re-run.  The window comes back as it was
left: a run saved maximized reopens maximized (a save while
maximized records the window's normal geometry, not the maximized
client rect), and a restored or default window size is clamped to
the screen.  With no saved run it starts from
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
    QAbstractItemModel,
    QAbstractTableModel,
    QCoreApplication,
    QEvent,
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
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QFont,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QScreen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog as QFileDialog,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QTableView,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from potions.gui.calibration import (
    CalibrationSettings,
    _CalibrationOutcome,
    _CalibrationWorker,
    ask_calibration_settings,
)
from potions.gui.common import (
    HYDRO_THEME,
    METRIC_NOTES,
    RT_MODE_TINT,
)
from potions.gui.data import (
    DEFAULT_CSV_PATH,
    GuiInputData,
    GuiRunResults,
    PlotSelection,
    arr_to_hydro_results,
    arr_to_model,
    arr_to_results,
    default_plot_selection,
    fingerprint_network,
    load_input_data,
    prefill_mapping,
)
from potions.gui.data_dialog import ColumnMappingDialog
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
    project_direction,
    search_t_points,
    write_line_search_csv,
)
from potions.gui.model_pool import map_model_runs
from potions.gui.model_specs import (
    DEFAULT_MODEL_KEY,
    MODEL_SPECS,
    ModelSpec,
    get_model_spec,
)
from potions.gui.network_editor import (
    _preset_fingerprint_map,
    edit_network,
)
from potions.gui.network_specs import (
    DEFAULT_NETWORK_KEY,
    NETWORK_SPECS,
    get_network_spec,
)
from potions.gui.param_panel import ParamPanel
from potions.gui.plots import (
    build_objective_table,
    make_conc_discharge_figure,
    make_gradient_figure,
    make_hydrology_figure,
    make_line_search_figure,
    make_partitioning_figure,
    make_rate_figure,
    make_river_concentrations_figure,
    make_river_loads_figure,
    make_storage_figure,
    make_zone_concentrations_figure,
)
from potions.gui.state import (
    MODE_FULL,
    MODE_HYDRO,
    NETWORK_FILENAME,
    GuiState,
    RunHistoryEntry,
    SavedRun,
    append_run_history,
    list_run_dirs,
    load_saved_run,
    objectives_from_results,
    saved_run_model_key,
    saved_run_network_fingerprint,
    write_gui_state,
)

# ==== Constants ==== #
# The window title (the title row repeats it in a larger font); the
# selected model's display name fills the placeholder.
_TITLE_TEMPLATE: Final[str] = "East River calibration — {model}"


def _title_for(spec: ModelSpec) -> str:
    """
    The window title for a model spec.

    Args:
        spec: The selected model's spec.

    Returns:
        The title naming the model's display name.
    """
    return _TITLE_TEMPLATE.format(model=spec.display_name)

# The debounce interval (ms) between a parameter change and the
# re-run; a drag or a burst of keyboard steps coalesces into one.
_DEBOUNCE_MS: Final[int] = 400

# The interval (ms) at which the elapsed-time label refreshes while
# a run is in flight.
_ELAPSED_MS: Final[int] = 200

# The initial window size (px), clamped to the screen at startup (see
# `_available_screen_geometry`); a restored window geometry replaces it
# before the window is shown.
_INITIAL_SIZE: Final[tuple[int, int]] = (1360, 900)

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

# Squeeze floors (px) for the header's shrinkable items.  Each
# item's plain minimum hint is its full text width; summed, the
# row would pin the window's minimumSizeHint wider than a common
# screen — and the window manager honors that hint, so the window
# could never be opened or shrunk onto the screen.  With floors, a
# narrow window squeezes the row instead: a floored item clips
# under pressure, its full text staying available in its tooltip
# or (for the combos) their popups.
_TITLE_MIN_WIDTH: Final[int] = 140
_COMBO_MIN_WIDTH: Final[int] = 100
_CONTROL_MIN_WIDTH: Final[int] = 48

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

# The width of the checkbox inset the objective table's delegate
# draws in each cell's left edge (px).
_CHECKBOX_INSET: Final[int] = 24

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
def _available_screen_geometry() -> QRect | None:
    """
    The current primary screen's available area, if there is a screen.

    Returns:
        The primary screen's `availableGeometry()` (the desktop minus
        panels), or None when there is no screen to fit into: the
        platform reports no primary screen, or it is the Qt offscreen
        plugin, whose synthetic 800x800 "screen" is no display a
        window could usefully be kept on.  Callers then apply their
        geometry unclamped.
    """
    if QGuiApplication.platformName() == "offscreen":
        return None
    screen: QScreen | None = QApplication.primaryScreen()
    return screen.availableGeometry() if screen is not None else None


def _clamped_geometry(requested: QRect, available: QRect) -> QRect:
    """
    Fit a requested window geometry inside the screen's available area.

    Sizes larger than the area are capped to it; a position whose
    window would hang past the area's far edge is pulled back so the
    whole window sits inside.  This keeps a restored or default window
    on the screen even when its saved geometry came from a bigger
    monitor (or was recorded while the window was maximized on one).

    Args:
        requested: The geometry the window asks for.
        available: The screen area the window must fit in.

    Returns:
        The requested geometry with its size capped to the area and
        its position pushed inside it.
    """
    width: int = min(requested.width(), available.width())
    height: int = min(requested.height(), available.height())
    x: int = min(
        max(requested.x(), available.x()),
        available.x() + available.width() - width,
    )
    y: int = min(
        max(requested.y(), available.y()),
        available.y() + available.height() - height,
    )
    return QRect(x, y, width, height)


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
        column_widths: The table's per-column rendered section
            sizes (what the view actually draws after
            re-resizing to contents).
        row_heights: The table's per-row rendered section sizes.
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
        model_key: The key of the model spec the run used; the
            history dropdown lists only the current model's runs.
    """

    label: str
    run_dir: Path | None
    param_values: dict[str, float] | None
    results: GuiRunResults | None
    mode: str | None
    model_key: str = DEFAULT_MODEL_KEY


# The fingerprint of the default network; None until first use, so
# building and hashing the network is paid at most once (and only
# when a saved run without a recorded fingerprint is actually
# matched, which an interactive session usually never does).
_DEFAULT_NETWORK_FINGERPRINT: str | None = None


def _default_network_fingerprint() -> str:
    """
    The fingerprint of the default reaction network.

    Built lazily once per process; it is the fingerprint every
    saved run without a recorded one ran on, since the GUI has
    always started on the default network.

    Returns:
        The SHA-256 fingerprint of the default network.
    """
    global _DEFAULT_NETWORK_FINGERPRINT
    if _DEFAULT_NETWORK_FINGERPRINT is None:
        _DEFAULT_NETWORK_FINGERPRINT = fingerprint_network(
            get_network_spec(DEFAULT_NETWORK_KEY).builder()
        )
    return _DEFAULT_NETWORK_FINGERPRINT


def _run_matches_network(run_dir: Path, fingerprint: str) -> bool:
    """
    Whether a saved run recorded the network ``fingerprint``.

    A run whose state file carries a network fingerprint matches
    on equality; a run that predates the field (or whose state is
    unreadable) is treated as a run of the default network (the
    spec's rule for pre-fingerprint runs).

    Args:
        run_dir: The saved run's directory.
        fingerprint: The fingerprint to match against.

    Returns:
        Whether the run's recorded network is ``fingerprint``.
    """
    recorded: str | None = saved_run_network_fingerprint(run_dir)
    if recorded is not None:
        return recorded == fingerprint
    return fingerprint == _default_network_fingerprint()


def _run_network(run_dir: Path) -> pt.ReactionNetwork | None:
    """
    The network a saved run recorded in its ``network.yaml``.

    `_on_save_run` writes the file; this reads it back so a
    restore of an old run can switch the window to that run's
    network.

    Args:
        run_dir: The saved run's directory.

    Returns:
        The recorded network, or None when the file is absent (a
        pre-integration run) or fails to load or validate; the
        failure is reported on stderr, not raised, since a broken
        optional file must not block the restore.
    """
    path: Path = run_dir / NETWORK_FILENAME
    if not path.is_file():
        return None
    try:
        return pt.ReactionNetwork.from_file(str(path))
    except (OSError, ValueError) as error:
        print(f"Could not load {path}: {error}", file=sys.stderr)
        return None


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
        model_key: The key of the model spec the job ran; the window
            discards an outcome from another model.
    """

    error: str | None
    table: pd.DataFrame | None
    base_results: GuiRunResults | None
    base_x: np.ndarray
    mode: str
    csv_path: Path | None
    elapsed_seconds: float
    model_key: str
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
        model_key: The key of the model spec the job ran; the window
            discards a snapshot from another model.
    """

    table: pd.DataFrame
    base_x: np.ndarray
    mode: str
    csv_path: Path
    elapsed_seconds: float
    model_key: str


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
        data_path: The path of the record file each pool worker
            loads in its initializer.
        base_results: The base point's results when they are
            already up to date on screen (None to run the base
            point too).
        model_key: The key of the model spec the job runs.
        column_map: The record's role → column mapping; None for
            an auto-detected record.
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
        data_path: str,
        base_results: GuiRunResults | None,
        model_key: str = DEFAULT_MODEL_KEY,
        network: pt.ReactionNetwork | None = None,
        column_map: dict[str, str | None] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._base_x: np.ndarray = base_x
        self._steps: list[_Perturbation] = steps
        self._mode: str = mode
        self._data_path: str = data_path
        self._base_results: GuiRunResults | None = base_results
        self._model_key: str = model_key
        self._network: pt.ReactionNetwork | None = network
        self._column_map: dict[str, str | None] | None = column_map

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
                model_key=self._model_key,
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
            self._data_path,
            on_result=on_result,
            model_key=self._model_key,
            network=self._network,
            column_map=self._column_map,
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
                        model_key=self._model_key,
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
            model_key=self._model_key,
            failed_points=failed_points,
            total_points=step_total,
        )


# ==== Line search ==== #
@dataclass(frozen=True, slots=True)
class _LineSearchOutcome:
    """
    The outcome of one finished line search job.

    A successful job carries the searched objectives' value
    profile (one row per evaluated point: the step ``t`` as
    index, the searched value — the equal-weighted sum of the
    selected terms — and one column per selected term, plus
    every free parameter's value, as columns), the best point's
    parameter vector and results (so the window can adopt them),
    and the path of the CSV the profile was written to; a failed
    job carries the error instead, with the rest None.  The
    selected terms and the job's mode ride on every outcome: the
    window shows them in the completion status line and the
    page's title, and it needs the mode to tell whether the best
    point's results are up to date.

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
        terms: The objective terms the job searched, as
            ``"target: metric"`` labels, in the objective
            table's row order.
        mode: The run mode the job's points ran in (`MODE_FULL` or
            `MODE_HYDRO`).
        elapsed_seconds: The wall-clock duration of the job.
        model_key: The key of the model spec the job ran; the window
            discards an outcome from another model.
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
    terms: list[str]
    mode: str
    elapsed_seconds: float
    model_key: str
    failed_points: int = 0
    total_points: int = 0


class _LineSearchWorker(QThread):
    """
    One line search job on a worker thread.

    The job follows the steepest ascent of the equal-weighted sum
    of the selected objective terms from the base point: when the
    direction was not handed in, it is computed by the same
    central differences as the gradient job (summing the selected
    terms' rows), and the sum is then evaluated at a fixed number
    of points on the ascent line, all inside the current
    parameter bounds.  The point with the largest finite value is
    the best one (the searched metrics all have 1 as their
    optimal value, so maximizing the sum finds the best point).

    When the direction is computed, its perturbation points are
    modelled in a worker pool, then the search's grid is built: a
    fixed number of points, equally spaced over the interval of
    steps that keeps every parameter inside the current bounds,
    with the point nearest the base point snapped onto it.  The
    base point itself is not run again: its profile row reuses the
    results already on screen (the handed-in base results, or the
    job's own base run), and the pool runs only the other points
    of the grid.

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
        terms: The objective terms to search, as
            ``"target: metric"`` labels, in the objective
            table's row order.
        mode: The run mode (`MODE_FULL` or `MODE_HYDRO`) the
            points run in.
        data_path: The path of the record file each pool worker
            loads in its initializer.
        bounds: The current parameter bounds, keyed by parameter.
        base_results: The base point's results when they are
            already up to date on screen (None to run the base
            point too).
        model_key: The key of the model spec the job runs.
        column_map: The record's role → column mapping; None for
            an auto-detected record.
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
        terms: list[str],
        mode: str,
        data_path: str,
        bounds: dict[str, tuple[float, float]],
        base_results: GuiRunResults | None,
        model_key: str = DEFAULT_MODEL_KEY,
        network: pt.ReactionNetwork | None = None,
        column_map: dict[str, str | None] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._base_x: np.ndarray = base_x
        self._steps: list[_Perturbation] = steps
        self._direction: np.ndarray | None = direction
        self._terms: list[str] = terms
        self._mode: str = mode
        self._data_path: str = data_path
        self._bounds: dict[str, tuple[float, float]] = bounds
        self._base_results: GuiRunResults | None = base_results
        self._model_key: str = model_key
        self._network: pt.ReactionNetwork | None = network
        self._column_map: dict[str, str | None] | None = column_map

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
                terms=list(self._terms),
                mode=self._mode,
                elapsed_seconds=time.perf_counter() - started,
                model_key=self._model_key,
            )
        finally:
            sys.stdout = real_stdout
        self.finished_line_search.emit(outcome)

    def _point_values(self, results: GuiRunResults) -> np.ndarray:
        """
        The job's selected objectives' values at one point's
        results, one entry per term.

        Args:
            results: One point's model results.

        Returns:
            One value per selected term, in the terms' order; an
            entry is NaN when the run did not report that term.
        """
        labels, vector = objective_vector(results)
        return np.array(
            [
                float(vector[labels.index(term)])
                if term in labels
                else float("nan")
                for term in self._terms
            ],
            dtype=np.float64,
        )

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
            terms=list(self._terms),
            mode=self._mode,
            elapsed_seconds=time.perf_counter() - started,
            model_key=self._model_key,
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
        the gradient job — and the direction is then the sum of
        the selected objectives' rows of their difference
        matrix, normalized to unit length and expanded to the
        parameter vector's space (a pinned parameter's component
        stays zero).

        The direction is then projected onto the feasible cone: a
        component that points out of a bound the base point sits
        on is dropped, and the surviving vector is renormalized,
        so the search only ever moves along a direction that stays
        inside the current bounds (a direction whose components are
        all blocked this way fails the job).  The search then
        evaluates the sum of the selected objectives at
        `_LINE_SEARCH_POINTS` points on the ascent line
        ``x + t * direction``, ``t`` equally spaced across the
        interval that keeps every parameter inside its bounds, with
        the point nearest the base point snapped onto it: the base
        point's row reuses the results the job already has for it
        (the handed-in base results, or the job's own base run)
        instead of being re-run, and the pool runs only the other
        points.  The point with the largest finite value is the
        best.

        Each completed point is reported as a (done, total) step
        count, for the window's progress bar: the base point when
        it was run, then each direction point when the direction is
        computed, then the search's pool points as they complete
        and the base point's row once the pool has finished.

        A failed base or direction point — including a failed
        pool initializer, which fails every point its worker
        receives — fails the whole job with the first failure's
        message, in the points' input order.  A failed search
        point does not fail the job: its entry in the profile is
        NaN (the missing value) and the failure is counted, so
        the window can report it in the completion status; the
        search for the best point skips NaN entries, so a job
        whose pool points all failed still reports the base
        row's value, which is the t = 0 row reusing the base
        results; the error is raised only when no grid row — the
        base row included — reports a value.

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
                Raised when the base point did not report every
                selected objective, when their summed differences
                carry no direction, when the ascent direction
                points out of the current bounds at every free
                parameter that can move (no feasible direction),
                or when no point on the ascent line lies inside
                the current bounds, or when no point of the
                search grid — the base row included — reports a
                value.
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
                self._data_path,
                on_result=on_phase1_result,
                model_key=self._model_key,
                network=self._network,
                column_map=self._column_map,
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
        # Project the direction onto the feasible cone before the
        # interval is computed: a component that points out of a
        # bound the base point sits on would otherwise collapse the
        # feasible interval onto the base point, so the search
        # could never move.
        direction = project_direction(
            self._base_x, direction, self._steps, self._bounds
        )

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
        t_points: np.ndarray = search_t_points(t_lo, t_hi, _LINE_SEARCH_POINTS)
        search_labels: list[str] = [f"t = {t:.4f}" for t in t_points]
        # The grid point that lands exactly on the base point (t = 0)
        # is not re-run in the pool: its profile row reuses the
        # results the job already has for the base point.
        zero_positions: set[int] = {
            index for index, t in enumerate(t_points) if t == 0.0
        }
        pool_positions: list[int] = [
            index
            for index in range(len(t_points))
            if index not in zero_positions
        ]
        pool_points: list[np.ndarray] = [
            self._base_x + t_points[index] * direction for index in pool_positions
        ]

        def on_pool_result(
            index: int, payload: GuiRunResults | str
        ) -> None:
            nonlocal done
            done += 1
            position: int = pool_positions[index]
            self.progress_line.emit(
                f"Line search run {done}/{total}: {search_labels[position]}"
            )
            self.progress_step.emit(done, step_total)

        pool_payloads: list[GuiRunResults | str] = (
            map_model_runs(
                pool_points,
                self._mode,
                self._data_path,
                on_result=on_pool_result,
                model_key=self._model_key,
                network=self._network,
                column_map=self._column_map,
            )
            if pool_points
            else []
        )

        # One entry per grid point, in the grid's order: the base
        # results' values fill the t = 0 row, the pool's payloads
        # (in the pool's input order) fill the rest.
        point_results: list[GuiRunResults | str] = []
        for _t in t_points:
            point_results.append(base_results)
        # The loop variable differs from the profile loop's
        # position: mypy's no-redef rule bars one function scope
        # defining a local name twice, loop variable included.
        for pool_index, grid_position in enumerate(pool_positions):
            point_results[grid_position] = pool_payloads[pool_index]
        # The base point's row was not run in the pool, so its
        # progress step is reported once the pool has finished.
        # The loop variable differs from the splice loop's: mypy's
        # no-redef rule bars one function scope defining a local
        # name twice, loop variable included.
        for zero_position in sorted(zero_positions):
            done += 1
            self.progress_line.emit(
                f"Line search run {done}/{total}: "
                f"{search_labels[zero_position]}"
            )
            self.progress_step.emit(done, step_total)

        failed_points: int = 0
        term_values: list[list[float]] = []
        for payload in point_results:
            if isinstance(payload, str):
                failed_points += 1
                term_values.append([float("nan")] * len(self._terms))
            else:
                term_values.append(
                    [float(value) for value in self._point_values(payload)]
                )
        all_term_values: np.ndarray = np.array(term_values, dtype=np.float64)
        # A point's value is the sum of the selected objectives'
        # values: a point whose run failed or that missed a term
        # carries a missing value (its missing term keeps the sum
        # missing).
        values_array: np.ndarray = np.where(
            np.isfinite(all_term_values).all(axis=1),
            np.sum(all_term_values, axis=1),
            float("nan"),
        )
        if not np.isfinite(values_array).any():
            raise ValueError(
                f"Every point on the ascent line failed to report a "
                f"value for {', '.join(self._terms)}."
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
        # One column per selected term, in the terms' order (the
        # objective table's row order).
        for column_index, term in enumerate(self._terms):
            columns[term] = all_term_values[:, column_index]
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
            terms=list(self._terms),
            mode=self._mode,
            elapsed_seconds=time.perf_counter() - started,
            model_key=self._model_key,
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
        selected objectives' equal-weighted sum, from its
        direction points' central differences.

        The direction's points — every free parameter perturbed
        up and down by 1% of its range (clamped to the range's
        ends), deduplicated against the base point, like the
        gradient job — were already run in the job's first pool
        phase; their results, together with the base point's,
        are `results`.  This method reads them only, so it runs
        no points and reports no progress of its own.  The
        direction is the sum of the selected objectives' rows of
        the difference matrix, normalized to unit length and
        expanded to the parameter vector's space (a pinned
        parameter's component stays zero).

        Args:
            base_results: The base point's results (its objective
                vector carries the selected terms' labels).
            results: The first phase's results, keyed by each
                point's value tuple (the base point's included).

        Returns:
            The unit-length direction vector, in the parameter
            vector's order.

        Raises:
            ValueError:
                Raised when the base point did not report every
                selected objective, when the selected objectives'
                summed differences are not all finite, or when
                they are all zero (no direction).
        """
        base_labels, _ = objective_vector(base_results)
        missing: list[str] = [
            term for term in self._terms if term not in base_labels
        ]
        if missing:
            raise ValueError(
                "The base point did not report every selected "
                f"objective: {', '.join(self._terms)}."
            )
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
        rows: list[int] = [base_labels.index(term) for term in self._terms]
        summed: np.ndarray = np.sum(matrix[rows], axis=0)
        if not np.isfinite(summed).all():
            raise ValueError(
                "The selected objectives have a missing difference at "
                f"the base point, so the direction cannot be computed: "
                f"{', '.join(self._terms)}."
            )
        norm: float = float(np.linalg.norm(summed))
        if norm == 0.0:
            raise ValueError(
                "The selected objectives are flat at the base point: "
                "their summed differences are all zero, so the line "
                f"search has no direction: {', '.join(self._terms)}."
            )
        return expand_direction(summed / norm, self._steps, self._bounds)


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
            index, the searched value and one column per
            selected term, plus every free parameter's value, as
            columns).
        terms: The objective terms the job searched, as
            ``"target: metric"`` labels, shown in the page's
            title.
        mode: The run mode the job's points ran in (`MODE_FULL` or
            `MODE_HYDRO`), shown in the page's title.
        csv_path: The path of the written profile CSV, quoted in
            the page's caption.
        elapsed_seconds: The wall-clock duration of the job, shown
            in the completion status line.
        model_key: The key of the model spec the job ran; the window
            discards a snapshot from another model.
    """

    profile: pd.DataFrame
    terms: list[str]
    mode: str
    csv_path: Path
    elapsed_seconds: float
    model_key: str


# ==== Objective table ==== #
class _ObjectiveTableModel(QAbstractTableModel):
    """
    A read-only item model over the per-run objective-function
    table, for the panel left of the parameter panel.

    Cells render their values at three decimals with a placeholder
    for a missing or NaN value; the row and column headers are the
    table's own (metric labels and target names).  The cells also
    carry the checkboxes that pick the line search's objectives:
    the model stores the checked terms (None for an unset
    selection, which searches every selectable term) and the
    selection's delegate and viewport filter (a left press on a
    selectable cell toggles its term) keep it and the view in
    sync.

    Args:
        parent: The parent object; defaults to None.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._table: pd.DataFrame = build_objective_table(
            pd.Series(dtype=float), None
        )
        # The checked objective terms, as ``"target: metric"``
        # labels; None for an unset selection (no checkbox
        # checked, the search falls back to every selectable
        # term).
        self._selected: frozenset[str] | None = None

    def set_table(self, table: pd.DataFrame) -> None:
        """
        Replace the underlying table, notifying the view.

        Args:
            table: The new per-run objective table.
        """
        self.beginResetModel()
        self._table = table
        self.endResetModel()

    def set_selected(self, terms: frozenset[str] | None) -> None:
        """
        Remember the checked cells and repaint the view.

        Args:
            terms: The checked objective terms, as
                ``"target: metric"`` labels; None for an unset
                selection (no checkbox checked).
        """
        self._selected = terms
        if self._table.empty:
            return
        first: QModelIndex = self.index(0, 0)
        last: QModelIndex = self.index(
            self.rowCount() - 1, self.columnCount() - 1
        )
        # The signal, not the shadowed protected method, is the
        # callable binding in PySide6.
        self.dataChanged.emit(first, last)

    def term_at(self, index: QModelIndex) -> str:
        """
        The objective term of one cell.

        Args:
            index: A valid cell index of the table.

        Returns:
            The cell's term, the column's target name and the
            row's metric label joined as ``"target: metric"``.
        """
        return f"{self._table.columns[index.column()]}: {self._table.index[index.row()]}"

    def is_selectable(self, term: str) -> bool:
        """
        Whether one objective term can be checked.

        A term is selectable when its metric is one the line
        search can optimize and its cell carries a finite value in
        the current table.

        Args:
            term: The objective term, ``"target: metric"``.

        Returns:
            True when the term is a searchable metric with a
            finite value in the table's cell; False otherwise
            (malformed term, unknown cell, unsearchable metric, or
            a missing value).
        """
        parts: list[str] = term.split(": ", 1)
        if len(parts) != 2:
            return False
        target, metric = parts
        if metric not in _LINE_SEARCH_METRICS:
            return False
        try:
            cell: float = float(self._table.loc[metric, target])
        except KeyError:
            return False
        return bool(np.isfinite(cell))

    def is_selected(self, term: str) -> bool:
        """
        Whether one objective term is currently checked.

        Args:
            term: The objective term, ``"target: metric"``.

        Returns:
            True when the term is in the checked selection; an
            unset selection checks no term.
        """
        if self._selected is None:
            return False
        return term in self._selected

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
        """
        The display text of a cell: the value at three visible
        digits (scientific notation when the magnitude is large),
        or a placeholder.
        """
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
        return f"{float(cell):.3g}"

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = 0
    ) -> str | None:
        """The column (target) or row (metric) label of a header section."""
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._table.columns[section])
        return str(self._table.index[section])


class _ObjectiveSelectionDelegate(QStyledItemDelegate):
    """
    The objective table's item delegate: draws each cell's checkbox
    in the cell's left inset.

    The checkbox is the cell's selectable state, not the cell's
    content: a selectable term (a searchable metric with a finite
    value) shows an enabled checkbox that is checked when the term
    is in the model's selection, and a non-selectable term shows a
    disabled (greyed) unchecked one, so the user can tell at a
    glance which cells the line search can search.

    Args:
        parent: The parent object (the table's view); defaults to
            None.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    def sizeHint(
        self, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QSize:
        """The base size hint, widened by the checkbox inset."""
        hint: QSize = super().sizeHint(option, index)
        return QSize(hint.width() + _CHECKBOX_INSET, hint.height())

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        """
        Draw the cell, then the checkbox in its left inset.

        The base style's text is drawn from a copy of the cell's
        rect whose left edge starts at the inset, so the value text
        starts to the right of the checkbox and is never covered by
        it; the checkbox itself is still drawn over the cell's
        unshifted left inset.
        """
        shifted: QStyleOptionViewItem = QStyleOptionViewItem(option)
        shifted.rect = shifted.rect.adjusted(_CHECKBOX_INSET, 0, 0, 0)
        super().paint(painter, shifted, index)
        model: QAbstractItemModel | None = index.model()
        if not isinstance(model, _ObjectiveTableModel) or not index.isValid():
            return
        term: str = model.term_at(index)
        checkbox: QStyleOptionButton = QStyleOptionButton()
        # The checkbox occupies the cell's left inset, vertically
        # centered.
        checkbox.rect = QRect(
            option.rect.left(),
            option.rect.center().y() - 8,
            _CHECKBOX_INSET,
            16,
        )
        if model.is_selectable(term):
            # State_Enabled keeps the box drawn at full contrast.
            checkbox.state = (
                QStyle.StateFlag.State_On
                if model.is_selected(term)
                else QStyle.StateFlag.State_Off
            ) | QStyle.StateFlag.State_Enabled
        else:
            # Without State_Enabled the style draws the unchecked
            # box greyed out.
            checkbox.state = QStyle.StateFlag.State_Off
        QApplication.style().drawControl(
            QStyle.ControlElement.CE_CheckBox,
            checkbox,
            painter,
        )


class _ObjectiveSelectionFilter(QObject):
    """
    The objective table's viewport event filter: a left mouse press
    on a selectable cell toggles that cell's objective term.

    The filter watches only the table's viewport (cell positions
    are the viewport's own), maps the press position to a cell,
    and emits the cell's term when the cell is selectable; presses
    on non-selectable cells, on the headers, and releases emit
    nothing.

    Signals:
        toggled: Emitted with the term whose cell was pressed, as
            ``"target: metric"``.

    Args:
        table: The objective table whose viewport is watched.
        parent: The parent object; defaults to None.
    """

    toggled = Signal(str)

    def __init__(self, table: QTableView, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._table: QTableView = table

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            isinstance(event, QMouseEvent)
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            index: QModelIndex = self._table.indexAt(event.pos())
            model: QAbstractItemModel | None = index.model()
            if (
                index.isValid()
                and isinstance(model, _ObjectiveTableModel)
                and model.is_selectable(model.term_at(index))
            ):
                self.toggled.emit(model.term_at(index))
        return False


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


# ==== Maximize chord ==== #
class _MaximizeChordFilter(QObject):
    """
    Detect the Home+Up key chord on one window and toggle its maximize.

    Qt shortcuts cannot express a two-key chord: ``Home+Up`` is not a
    key, adding the key codes parses to a wrong sequence, and a
    two-entry *sequence* shortcut never fires when a focused child
    consumes the first key.  So the chord is detected from raw key
    events: the filter is installed on the application (application
    filters run before any widget's own handling, unlike window
    actions, which a focused child's shortcut handling can beat),
    watches only the events delivered to its window's own widget
    tree, and consumes only the *completing* press of the pair — a
    lone Home or a lone Up reaches its widget untouched.

    The chord fires on the second key pressed (either order) with no
    modifier held; auto-repeat and further presses do not re-fire
    until both keys have been released, so holding the chord down
    toggles exactly once.
    """

    _CHORD_KEYS: Final[frozenset[Qt.Key]] = frozenset(
        {Qt.Key.Key_Home, Qt.Key.Key_Up}
    )

    def __init__(self, window: QMainWindow) -> None:
        """
        Args:
            window: The window whose maximize the chord toggles.  The
                filter is parented to it, so Qt removes it from the
                application when the window is destroyed.
        """
        super().__init__(window)
        self._window: QMainWindow = window
        self._held: set[Qt.Key] = set()
        self._fired: bool = False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Consume the completing key press of the Home+Up chord.

        Args:
            watched: The object receiving the event — every object in
                the application, so the guards run before anything is
                touched: non-key events first, then only events aimed
                at a widget belonging to this filter's own window (a
                non-widget receiver like the shared style has no
                window to compare).
            event: The event being delivered.

        Returns:
            True when the event is the completing press of the chord
            and was consumed; False otherwise.
        """
        if event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False
        if not isinstance(watched, QWidget):
            return False
        if watched.window() is not self._window:
            return False
        assert isinstance(event, QKeyEvent)
        key: Qt.Key = event.key()
        if event.type() == QEvent.Type.KeyRelease:
            self._held.discard(key)
            if not self._CHORD_KEYS & self._held:
                # Both chord keys are up again: the chord may fire once more.
                self._fired = False
            return False
        if (
            key in self._CHORD_KEYS
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self._CHORD_KEYS & self._held
            and not self._fired
            and not event.isAutoRepeat()
        ):
            self._fired = True
            self._held.add(key)
            self._window._toggle_maximized()
            return True
        self._held.add(key)
        return False


# ==== Main window ==== #
class MainWindow(QMainWindow):
    """
    The main window of the East River lateral-model calibration GUI.

    The window stacks two non-collapsible sections: the figure
    panel (the model's figures, one visible at a time: the
    hydrology, storage, and partitioning figures, joined by the
    river concentrations, river loads, concentration vs.
    discharge, zone concentrations, and reaction rates figures in
    full mode, plus the gradient and line search pages when those
    jobs have finished) and, below it, a row
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
    geometry, maximized state, and splitter sizes) and builds the
    figures from the run's saved output, so no re-run is needed;
    the restored window opens maximized when its run was saved
    maximized, and its size is clamped to the screen.  Home+Up
    toggles the window's maximize.  With no saved run it starts
    from the default parameters and runs once in the background.

    Args:
        input_data: The assembled calibration inputs (the
            parameter bounds and the reaction network).
        data_path: The path of the record file (a CSV or XLSX)
            the session used, or None for the record shipped with
            the package; stored in the saved run state.
        column_map: The record's role → column mapping, or None
            when its roles auto-detected; stored in the saved run
            state.
        network_name:
            The display name of a custom network the window starts
            on (the file's stem for a ``--network`` start); None
            when the window starts on a preset.
        parent: The parent widget; defaults to None.
    """

    def __init__(
        self,
        input_data: GuiInputData,
        data_path: str | None = None,
        column_map: dict[str, str | None] | None = None,
        network_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._input_data: GuiInputData = input_data
        self._data_path: str | None = data_path
        self._column_map: dict[str, str | None] | None = column_map

        # The last results displayed in the figure and table.
        self._results: GuiRunResults | None = None
        # The run currently in flight (None while no run is in
        # flight; the reference is dropped when the worker ends,
        # before the deferred deletion removes the object).
        self._worker: (
            _RunWorker
            | _GradientWorker
            | _LineSearchWorker
            | _CalibrationWorker
            | None
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
        # The objective table cells the user checked as line search
        # objectives (as ``"target: metric"`` terms); None for an
        # unset selection, which searches every selectable term.
        # Restored from a saved run; never set by a run's results.
        self._objective_terms: frozenset[str] | None = None
        # Set when a model switch asked for a run but a worker was
        # still in flight; the run starts once that worker ends.
        self._pending_run_after_worker: bool = False
        # The network the window is running; the selector's
        # entries are keyed off its fingerprint.
        self._network: pt.ReactionNetwork = input_data.network
        # The display name of a custom (non-preset) network, or
        # None when the selector shows a preset entry.
        self._custom_network_name: str | None = network_name
        # The window's normal (pre-maximize) geometry: captured the
        # instant the window becomes maximized and updated whenever
        # the window moves or resizes as a normal window, so a *Save
        # run* while maximized records it instead of the maximized
        # client rect (see `_on_save_run`).  None until the first
        # resize or move.
        self._normal_geometry: QRect | None = None
        # Set by `_apply_saved_run` when the restored run was saved
        # maximized; consumed once by `showEvent`, which defers the
        # maximize until the window is fully shown.
        self._restore_maximized: bool = False

        self.setWindowTitle(_title_for(self._input_data.spec))
        # The default size is clamped to the screen so a window whose
        # saved history never existed still opens on the display.
        initial: tuple[int, int] = _INITIAL_SIZE
        screen_area: QRect | None = _available_screen_geometry()
        if screen_area is not None:
            initial = (
                min(initial[0], screen_area.width()),
                min(initial[1], screen_area.height()),
            )
        self.resize(*initial)

        self._build_ui()
        self._build_actions()
        # The Home+Up maximize chord (see `_MaximizeChordFilter`): a
        # window action cannot express a key chord, so the chord is
        # detected with an application-level filter that Qt removes
        # when the window is destroyed (it is parented to the window).
        self._chord_filter: _MaximizeChordFilter = _MaximizeChordFilter(self)
        application: QCoreApplication | None = QApplication.instance()
        if application is not None:
            application.installEventFilter(self._chord_filter)
        self._restore_or_start()
        # The restored mode (or the default) is final by now, so the
        # window can take it as its background tint.
        self._set_mode_tint(self._current_mode())

    # ==== Building ==== #
    def _build_ui(self) -> None:
        """Assemble the header row, the figure, the lower row, and the progress strip."""
        self._build_menus()
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

    def _build_menus(self) -> None:
        """
        Build the menu bar's menus: fixed chrome at the window's top.

        The File menu carries the data- and network-sourcing commands
        that used to sit as header buttons: menu items are the
        conventional home for file-ish commands, and menu items can
        carry Alt mnemonics (a text-only toolbar would print an `&`
        literally, so toolbar actions never get them).  Later builders
        add further menus to this bar; this method builds the bar's
        first menu and the actions inside it.
        """
        self._file_menu: QMenu = self.menuBar().addMenu("&File")

        self._load_data_action: QAction = QAction("&Load data…", self)
        self._load_data_action.setToolTip(
            "Load a CSV or XLSX record and switch to it.  The "
            "dialog maps the file's columns to the record's roles."
        )
        self._load_data_action.triggered.connect(self._on_load_data)
        self._file_menu.addAction(self._load_data_action)

        self._load_network_action: QAction = QAction(
            "&Load network file…", self
        )
        self._load_network_action.setToolTip(
            "Run the network of a YAML file.  A file whose network "
            "matches a preset selects that preset's entry."
        )
        self._load_network_action.triggered.connect(self._on_load_network)
        self._file_menu.addAction(self._load_network_action)

        self._save_network_action: QAction = QAction(
            "Save network &as…", self
        )
        self._save_network_action.setToolTip(
            "Write the network currently running to a YAML file."
        )
        self._save_network_action.triggered.connect(self._on_save_network)
        self._file_menu.addAction(self._save_network_action)

        self._edit_network_action: QAction = QAction("&Edit network…", self)
        self._edit_network_action.setToolTip(
            "Open the network editor.  Applying the dialog switches "
            "the window to the edited network."
        )
        self._edit_network_action.triggered.connect(self._on_edit_network)
        self._file_menu.addAction(self._edit_network_action)

    def _build_header(self) -> QWidget:
        """
        Build the header row: title, running indicator, status, and
        buttons.

        The row carries the fixed-size running indicator (a
        `_SpinnerLabel`) next to the title: its footprint does not
        change with the job state, so the window's minimum width is
        the same whether or not a job is in flight (the remaining
        job indicators live in the bottom strip).  The row's items
        are compressible down to squeeze floors (see
        `_TITLE_MIN_WIDTH`, `_COMBO_MIN_WIDTH`, and
        `_CONTROL_MIN_WIDTH`; the status label reports its own
        small minimum, see `_StatusLabel`), so a narrow window
        squeezes the row — clipping text whose full content stays
        in tooltips and combo popups — instead of pushing the
        window's minimum past the screen.
        """
        self._title_label: QLabel = QLabel(_title_for(self._input_data.spec))
        font: QFont = QFont(self._title_label.font())
        font.setBold(True)
        font.setPointSize(max(9, font.pointSize() + 2))
        self._title_label.setFont(font)

        # The running-job indicator: hidden at rest, its fixed size
        # keeps the row's footprint constant either way.
        self._job_spinner: _SpinnerLabel = _SpinnerLabel()

        # The model selector: one entry per curated model spec, the
        # selected model's display name in the text and its spec key
        # in the item data.  A change reloads the inputs and resets
        # the panel (see `_on_model_changed`).
        self._model_combo: QComboBox = QComboBox()
        self._model_combo.setToolTip(
            "Switch the calibrated model.  The parameters, bounds, "
            "zones, figures, and history follow the selected model; "
            "switching resets the parameter values to the new "
            "model's defaults."
        )
        for key, spec in MODEL_SPECS.items():
            self._model_combo.addItem(spec.display_name, key)
        self._model_combo.setCurrentIndex(
            self._model_combo.findData(self._input_data.spec.key)
        )
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)

        # The network selector: one entry per preset (the preset's
        # display name in the text, its key in the item data) plus,
        # when the window runs a network that is not one of the
        # presets, a single `custom: <name>` entry (the network's
        # fingerprint in the item data).  A change switches the
        # network (see `_on_network_changed`).
        self._network_combo: QComboBox = QComboBox()
        self._network_combo.setToolTip(
            "Switch the reaction network.  The measured species, "
            "the figures, the history, and the saved outputs "
            "follow the selected network; switching resets the "
            "parameters to the new network's defaults.  Use "
            "*Load network file…*, *Save network as…*, and *Edit "
            "network…* beside it to work with a network of your "
            "own."
        )
        for net_key, net_spec in NETWORK_SPECS.items():
            self._network_combo.addItem(net_spec.display_name, net_key)
        self._network_combo.setCurrentIndex(
            self._network_combo.findData(DEFAULT_NETWORK_KEY)
        )
        self._network_combo.currentIndexChanged.connect(
            self._on_network_changed
        )

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
            "Follow the steepest ascent of the selected objective "
            "functions' equal-weighted sum from the current "
            "parameters: check the cells of the objective table to "
            "select any combination of its KGE, LogKGE, NSE, or "
            "LogNSE values, and the job evaluates the sum at 25 "
            "points on the ascent line, all inside the current "
            "parameter bounds, and adopts the point with the best "
            "value (the searchable metrics have 1 as their optimal "
            "value, so the search maximizes the sum; if no cell "
            "has ever been checked, every selectable term is "
            "searched, but unchecking every checked cell refuses "
            "the search).  "
            "When a gradient job has already run at these "
            "parameters its matrix supplies the direction; "
            "otherwise the direction is computed by the same "
            "central differences.  The value profile is written to "
            "a CSV in the results directory and drawn as an extra "
            "page of the figure panel."
        )

        self._calibrate_button: QPushButton = QPushButton("Calibrate")
        self._calibrate_button.clicked.connect(self._on_calibrate_clicked)
        self._calibrate_button.setEnabled(False)
        self._calibrate_button.setToolTip(
            "Optimize the free hydrology parameters with differential "
            "evolution against the checked objective terms (or every "
            "selectable term when none is checked).  A parameter "
            "whose range has no width is pinned at its current value.  "
            "The best run is adopted into the panel and saved.  "
            "Available in hydrology-only mode."
        )

        self._reset_button: QPushButton = QPushButton("Reset bounds")
        self._reset_button.clicked.connect(self._on_reset_bounds)

        self._save_button: QPushButton = QPushButton("Save run")
        self._save_button.clicked.connect(self._on_save_run)
        self._save_button.setEnabled(False)

        # Squeeze floors for the row's shrinkable items (see the
        # `_TITLE_MIN_WIDTH` group above): without them every item
        # reports its full text width as its minimum and the row
        # would pin the window's minimumSizeHint wider than a
        # common screen.  Clipped items keep their full text
        # available — the combos in their popups, the checkboxes
        # and buttons in tooltips, falling back to the item's own
        # text where a button carries no more specific one.
        self._title_label.setMinimumWidth(_TITLE_MIN_WIDTH)
        self._title_label.setToolTip(self._title_label.text())
        combos: tuple[QComboBox, ...] = (
            self._model_combo,
            self._network_combo,
        )
        for combo in combos:
            combo.setMinimumWidth(_COMBO_MIN_WIDTH)
        controls: tuple[QPushButton | QCheckBox, ...] = (
            self._mode_checkbox,
            self._save_history_checkbox,
            self._auto_run_checkbox,
            self._run_button,
            self._gradients_button,
            self._line_search_button,
            self._calibrate_button,
            self._reset_button,
            self._save_button,
        )
        for control in controls:
            control.setMinimumWidth(_CONTROL_MIN_WIDTH)
            if not control.toolTip():
                control.setToolTip(control.text())

        layout: QHBoxLayout = QHBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(self._title_label)
        layout.addWidget(self._job_spinner)
        layout.addWidget(self._model_combo)
        layout.addWidget(self._network_combo)
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
        layout.addWidget(self._calibrate_button)
        layout.addWidget(self._reset_button)
        layout.addWidget(self._save_button)
        self._sync_network_selector()
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
        # The cells' checkboxes draw on top of the values: the
        # delegate paints them in each cell's left inset, and the
        # viewport's event filter turns a left press on a
        # selectable cell into a toggle of its term (the window
        # stores the checked terms and the model repaints them).
        self._objective_table.setItemDelegate(
            _ObjectiveSelectionDelegate(self._objective_table)
        )
        self._objective_selection_filter: _ObjectiveSelectionFilter = (
            _ObjectiveSelectionFilter(self._objective_table, self)
        )
        self._objective_table.viewport().installEventFilter(
            self._objective_selection_filter
        )
        self._objective_selection_filter.toggled.connect(
            self._on_objective_toggled
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
        table's own measurements run through `_objective_panel_size`:
        the section sizes the view actually renders (the wider of
        the header text and the cell content, recomputed here since
        both headers are in `ResizeToContents` mode — a header-text-
        only measurement would clip any column or row whose cells
        are wider) plus both headers' own size hints and the frame
        width, so the scrollbars (held off) never have work to do.
        """
        table = self._objective_table
        # Re-resizing is the view's own measurement of what it will
        # render; it is a no-op on the headers' sections while the
        # model is still empty (the build-time call site).
        table.horizontalHeader().resizeSections()
        table.verticalHeader().resizeSections()
        width, height = _objective_panel_size(
            self._objective_heading.sizeHint(),
            [
                table.horizontalHeader().sectionSize(c)
                for c in range(self._objective_model.columnCount())
            ],
            [
                table.verticalHeader().sectionSize(r)
                for r in range(self._objective_model.rowCount())
            ],
            table.verticalHeader().sizeHint().width(),
            table.horizontalHeader().sizeHint().height(),
            table.frameWidth(),
            _OBJECTIVE_PANEL_MARGINS,
            _OBJECTIVE_PANEL_SPACING,
        )
        self._objective_panel.setFixedSize(width, height)

    def _build_param_panel(self) -> ParamPanel:
        """Build the parameter panel and connect its signals to the window."""
        self._param_panel: ParamPanel = ParamPanel(
            self._input_data,
            default_plot_selection(
                self._input_data.spec,
                network=self._input_data.network,
                window=self._input_data.window,
            ),
            self,
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
        """
        Register the Left / Right arrow actions that flip the figure pages.

        Maximizing is not an action here: a QAction shortcut cannot
        express the Home+Up key chord, which `_MaximizeChordFilter`
        detects from raw key events instead.
        """
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

        The latest saved run decides the window's model and
        network: a saved run of another curated model switches the
        window to it first, and a saved run whose recorded network
        differs from the window's (a run without a recorded
        fingerprint counts as one of the default network) switches
        the window to the run's network as well; neither switch
        starts a run, since the saved run is applied straight
        away.  The parameter panel's *History* dropdown is then
        filled with the current model's saved run directories of
        the running network, newest first, so the restored run is
        one of its entries; with a restorable latest run the
        window opens on that run's figures and parameters without
        a re-run, and otherwise (no run, or an unreadable one) it
        starts from the default parameters and runs the models
        once in the background.
        """
        runs: list[Path] = list_run_dirs()
        latest: Path | None = runs[0] if runs else None
        if latest is not None:
            key: str = saved_run_model_key(latest)
            if key != self._input_data.spec.key:
                index: int = self._model_combo.findData(key)
                if index >= 0:
                    self._model_combo.blockSignals(True)
                    self._model_combo.setCurrentIndex(index)
                    self._model_combo.blockSignals(False)
                    self._switch_model(key, start_run=False)
            recorded: str | None = (
                saved_run_network_fingerprint(latest)
                or _default_network_fingerprint()
            )
            if recorded != fingerprint_network(self._input_data.network):
                self._switch_network(
                    _run_network(latest)
                    or get_network_spec(DEFAULT_NETWORK_KEY).builder(),
                    start_run=False,
                )
        # The saved-run directories become entries of the window's
        # history (label and path only; their state is read on
        # demand), so a selection from the dropdown can restore and
        # overlay the same runs the dropdown shows.  Only the
        # current model's runs of the running network are listed.
        current_key: str = self._input_data.spec.key
        current_fingerprint: str = fingerprint_network(
            self._input_data.network
        )
        current_runs: list[Path] = [
            run
            for run in runs
            if saved_run_model_key(run) == current_key
            and _run_matches_network(run, current_fingerprint)
        ]
        self._history_items = [
            _HistoryItem(
                label=run.name,
                run_dir=run,
                param_values=None,
                results=None,
                mode=None,
                model_key=current_key,
            )
            for run in current_runs
        ]
        self._param_panel.set_history_items(
            [run.name for run in current_runs]
        )
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
        the saved-window bits: the checked objective terms (only the
        checked ones: the table cells re-check themselves against
        the restored results), the status note, the window geometry
        and splitter sizes, the *Save run history* checkbox, and —
        only when the run's record is the window's current one (a
        path mismatch means the session's data did not come from
        that run, so the mapping is ignored, not forced) — the
        session's column mapping.
        """
        # The checked cells are restored before the shared apply, so
        # the results refresh that follows intersects them with the
        # restored table's selectable cells.
        self._objective_terms = (
            frozenset(saved.objective_terms)
            if saved.objective_terms is not None
            else None
        )
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
            requested: QRect = QRect(
                saved.window_geometry.get("x", 0),
                saved.window_geometry.get("y", 0),
                saved.window_geometry.get("width", _INITIAL_SIZE[0]),
                saved.window_geometry.get("height", _INITIAL_SIZE[1]),
            )
            screen_area: QRect | None = _available_screen_geometry()
            self.setGeometry(
                requested
                if screen_area is None
                else _clamped_geometry(requested, screen_area)
            )
        # The maximized state is restored through the window system
        # (see `showEvent`), not through the clamped geometry above:
        # the saved geometry is the run's *normal* geometry.
        self._restore_maximized = saved.window_maximized
        if saved.splitter_sizes is not None:
            self._splitter.setSizes(saved.splitter_sizes[:2])
        # The checkbox state is part of the saved state; a state
        # saved before the toggle existed restores it off.
        self._save_history_checkbox.setChecked(saved.save_history)
        # A saved mapping applies only to the record it was read
        # from: a run saved on another record (or the shipped one,
        # which reads by auto-detection) must not rename the
        # columns of the window's current record.
        if saved.column_map is not None and saved.csv_path == self._data_path:
            self._column_map = saved.column_map

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

    def _update_calibrate_button(self) -> None:
        """
        Sync the calibrate button with the run mode and the job state.

        Calibration optimizes only the hydrology model, so the button
        is available only in hydrology-only mode and only when no job
        is in flight.
        """
        busy: bool = self._worker is not None and self._worker.isRunning()
        self._calibrate_button.setEnabled(
            not busy and self._current_mode() == MODE_HYDRO
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
        self._calibrate_button.setEnabled(False)
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

        A model switch made while this worker was in flight asked
        for a run of the new model (see `_switch_model`): that run
        starts now that the window is free, or the run button lights
        up when auto-run is off.
        """
        self._worker = None
        if not self._pending_run_after_worker:
            return
        self._pending_run_after_worker = False
        if self._auto_run_checkbox.isChecked():
            self._start_run(self._current_x())
        else:
            self._run_button.setEnabled(True)
            self._gradients_button.setEnabled(True)
            self._line_search_button.setEnabled(True)
            self._update_calibrate_button()

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
        # re-disables them via `_start_run`); the calibrate button
        # follows the run mode.
        self._gradients_button.setEnabled(True)
        self._line_search_button.setEnabled(True)
        self._update_calibrate_button()
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
        self._calibrate_button.setEnabled(False)
        self._status("Computing gradients…")

        worker: _GradientWorker = _GradientWorker(
            x,
            steps,
            self._current_mode(),
            self._data_path or str(DEFAULT_CSV_PATH),
            base_results,
            self._input_data.spec.key,
            self._input_data.network,
            self._column_map,
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
        if outcome.model_key != self._input_data.spec.key:
            # The model changed mid-job: the outcome belongs to the
            # model the window no longer shows.  Drop it and restore
            # the job buttons the switch left disabled.
            self._gradients_button.setEnabled(True)
            self._line_search_button.setEnabled(True)
            self._update_calibrate_button()
            self._update_run_button()
            return
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
                model_key=outcome.model_key,
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
        self._update_calibrate_button()

    # ==== Line search ==== #
    def _on_line_search_clicked(self) -> None:
        """
        Start a line search of the checked objectives from the
        *Line search* button.

        The search's objectives are the cells of the objective
        table the user checked (any combination of its KGE, LogKGE,
        NSE, and LogNSE values); nothing checked means every
        selectable cell.  A job already in flight takes precedence:
        a click during a job's run is ignored silently, since a job
        cannot be interrupted and the search itself reports the
        refusal with a message.
        """
        if self._worker is not None and self._worker.isRunning():
            return
        self._start_line_search(self._objective_terms or frozenset())

    def _on_objective_toggled(self, term: str) -> None:
        """
        Toggle one objective table cell's term in the line search's
        selection.

        The viewport's event filter emits the term of the
        selectable cell a left press landed on: the first check of
        the session turns an unset selection into that single term,
        and each further check toggles the term's membership.  The
        selection is the user's — no re-run is triggered.

        Args:
            term: The pressed cell's term, ``"target: metric"``.
        """
        if not self._objective_model.is_selectable(term):
            return
        if self._objective_terms is None:
            terms: frozenset[str] = frozenset({term})
        else:
            terms = self._objective_terms ^ {term}
        self._objective_terms = terms
        self._objective_model.set_selected(terms)

    def _selectable_objective_terms(self) -> frozenset[str]:
        """
        The objective terms the line search can search now.

        The selectable cells of the current objective table (a
        searchable metric with a finite value), narrowed to the run
        mode: a hydro run models streamflow only, so only the
        streamflow terms are selectable in hydro mode.

        Returns:
            The selectable terms, as ``"target: metric"`` labels.
        """
        prefix: str = (
            "Streamflow:" if self._current_mode() == MODE_HYDRO else ""
        )
        terms: set[str] = set()
        for row in range(self._objective_model.rowCount()):
            for column in range(self._objective_model.columnCount()):
                index: QModelIndex = self._objective_model.index(row, column)
                term: str = self._objective_model.term_at(index)
                if term.startswith(prefix) and self._objective_model.is_selectable(
                    term
                ):
                    terms.add(term)
        return frozenset(terms)

    def _sync_objective_selection(self, selectable: frozenset[str]) -> None:
        """
        Keep the checked terms a subset of the selectable ones.

        An explicit selection is intersected with the terms
        selectable now, so a term that leaves the table (a mode
        switch) or loses its finite value (a new run's results)
        drops out of it, while an unset selection stays unset; the
        model's checkboxes repaint either way.

        Args:
            selectable: The objective terms selectable now.
        """
        if self._objective_terms is not None:
            self._objective_terms &= selectable
        self._objective_model.set_selected(self._objective_terms)

    def _start_line_search(self, terms: frozenset[str]) -> None:
        """
        Start a line search of the selected objectives'
        equal-weighted sum.

        The search follows the steepest ascent of the sum of the
        selected objective terms from the current parameters: when
        a gradient job has already finished at these parameters (in
        the same mode, with the same free parameters) and its table
        reports every selected term, its matrix supplies the
        direction (the summed rows, un-normalized) for free;
        otherwise the direction is computed by the same central
        differences as the gradient job, summing the selected
        terms' rows.  The direction is then projected onto the
        feasible cone (a component that points out of a bound the
        base point sits on is dropped), and the sum is evaluated at
        25 points on the ascent line, all inside the current
        parameter bounds, with the grid's point nearest the base
        point snapped onto it (its row reuses the base results
        rather than re-running the point).  The point with the
        best (largest) value is adopted when it strictly beats the
        base point's value, or when the base value is missing
        (possible only when the direction was handed in); when the
        base value is finite and the best does not beat it, the
        search found no improvement, leaves the parameters
        unchanged, and says so in the status line.

        The search's objectives are the checked cells of the
        objective table; an unset selection (nothing checked)
        searches every selectable cell.  The run indicators show
        the job's progress like a model run's; a job cannot be
        interrupted.

        The refusals, in order: a job already in flight (a model
        run or a gradient job) takes precedence and is reported
        with a message, since a job cannot be interrupted; an
        empty table has no values to search yet; no selectable
        term is checked; no free parameter to search.
        """
        if self._worker is not None and self._worker.isRunning():
            self._status(
                "A job is already in flight: it cannot be interrupted, "
                "so the line search was not started."
            )
            return
        table: pd.DataFrame = self._objective_model._table
        if table.empty:
            self._status(
                "No objective values to search yet: run the model first."
            )
            return
        # An unset selection searches every selectable term; a set
        # one searches exactly the checked terms.
        effective: frozenset[str] = (
            terms
            if self._objective_terms is not None
            else self._selectable_objective_terms()
        )
        if not effective:
            self._status(
                "No objectives are selected: check at least one cell of "
                "the objective table."
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
        # The selected terms in the objective table's row order —
        # the order the worker evaluates and reports them in.
        terms_list: list[str] = [
            f"{target}: {metric}"
            for metric in table.index
            for target in table.columns
            if f"{target}: {metric}" in effective
        ]
        direction: np.ndarray | None = None
        snapshot: _GradientSnapshot | None = self._gradient_state
        if (
            snapshot is not None
            and snapshot.mode == self._current_mode()
            and np.array_equal(snapshot.base_x, x)
            and len(snapshot.table.columns) == len(steps)
        ):
            try:
                rows: list[np.ndarray] = [
                    snapshot.table.loc[term].to_numpy(dtype=np.float64)
                    for term in terms_list
                ]
                direction = expand_direction(
                    np.sum(np.vstack(rows), axis=0), steps, bounds
                )
            except KeyError:
                # A selected term the snapshot's table lacks:
                # compute the direction fresh instead.
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
        self._calibrate_button.setEnabled(False)
        if len(terms_list) == 1:
            self._status(f"Line search in progress: {terms_list[0]} …")
        else:
            self._status(
                f"Line search in progress: "
                f"{len(terms_list)} selected objectives …"
            )
        self._worker = _LineSearchWorker(
            x,
            steps,
            direction,
            terms_list,
            self._current_mode(),
            self._data_path or str(DEFAULT_CSV_PATH),
            bounds,
            base_results,
            self._input_data.spec.key,
            self._input_data.network,
            self._column_map,
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
        always kept as the figure panel's extra page, and its
        completion is reported in the status bar with the path of
        the written CSV.  When the best point's value strictly
        beats the base point's value (the t = 0 row) — or the base
        value is missing, possible only on the
        handed-in-direction path since the computed path validates
        the base first — the best point's parameters and results
        are adopted into the window (the panel is restored to the
        best point's values and the figures and table refresh),
        and the status reports the best step and value.  When the
        base value is finite and the best does not exceed it,
        nothing is adopted: the parameters and the results on
        screen stay exactly as they were, and the status says no
        improvement was found.
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
        if outcome.model_key != self._input_data.spec.key:
            # The model changed mid-job: the outcome belongs to the
            # model the window no longer shows.  Drop it (the job
            # buttons were already restored above).
            self._update_calibrate_button()
            self._update_run_button()
            return
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
            self._update_calibrate_button()
            return

        # The best point is the profile's largest finite value.
        values: np.ndarray = outcome.profile["value"].to_numpy(
            dtype=np.float64
        )
        best_index: int = int(np.nanargmax(values))
        best_t: float = float(outcome.profile.index[best_index])
        best_value: float = float(values[best_index])
        # The base point's value is the value at the t = 0 row.  The
        # search found no improvement when the best value does not
        # strictly beat it: the searched metrics all have 1 as their
        # optimal value, so a best that equals the base means the
        # search could not do better than where it started.
        t_values: np.ndarray = outcome.profile.index.to_numpy(
            dtype=np.float64
        )
        zero_mask: np.ndarray = t_values == 0.0
        base_value: float | None = (
            float(values[int(zero_mask.argmax())])
            if zero_mask.any()
            else None
        )

        # The finished profile is kept as the figure panel's page in
        # both outcomes, so it is set before the adoption decision.
        self._line_search_state = _LineSearchSnapshot(
            profile=outcome.profile,
            terms=list(outcome.terms),
            mode=outcome.mode,
            csv_path=outcome.csv_path,
            elapsed_seconds=outcome.elapsed_seconds,
            model_key=outcome.model_key,
        )
        if (
            base_value is not None
            and np.isfinite(base_value)
            and best_value <= base_value
        ):
            # No improvement: the parameters and the results on
            # screen stay exactly as they were; only the profile
            # page and the written CSV are recorded, and the
            # completion says the parameters are unchanged.
            self._refresh_from_results()
            # If branches share the function's scope in Python, so
            # this local cannot reuse the adoption branch's name
            # (mypy's no-redef rule); the distinct name says which
            # branch's status it carries.
            no_improvement_message: str = (
                f"Line search finished in {outcome.elapsed_seconds:.1f} s: "
                "no improvement found (the best value equals the base "
                "value); the parameters are unchanged and the profile "
                f"was saved to {outcome.csv_path}."
            )
            if outcome.failed_points > 0:
                no_improvement_message += (
                    f" {outcome.failed_points} of {outcome.total_points} "
                    "points failed; their values are shown as missing."
                )
            self._status(no_improvement_message)
            up_to_date = self._results_are_up_to_date()
            self._save_button.setEnabled(up_to_date)
            self._run_button.setEnabled(not up_to_date)
            self._update_calibrate_button()
            return

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
        self._refresh_from_results()
        if len(outcome.terms) == 1:
            best_text: str = (
                f"({outcome.terms[0]} = {best_value:.3g}); the best "
                f"point's parameters are in the panel and the profile "
                f"was saved to {outcome.csv_path}."
            )
        else:
            best_text = (
                f"(sum of {len(outcome.terms)} objectives = "
                f"{best_value:.3g}); the best point's parameters are "
                f"in the panel and the profile was saved to "
                f"{outcome.csv_path}."
            )
        message: str = (
            f"Line search finished in {outcome.elapsed_seconds:.1f} s: "
            f"best t = {best_t:.4f} {best_text}"
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
        self._update_calibrate_button()

    # ==== Calibration ==== #
    def _on_calibrate_clicked(self) -> None:
        """
        Ask for the calibration's options and start the job.

        The job optimizes the free hydrology parameters against the
        checked objective terms (every selectable term when none is
        checked).  It refuses, with a message, when a job is already
        in flight, when the window is not in hydrology-only mode,
        when the table has no values yet, when no term is
        selectable, or when no hydrology parameter is free.
        """
        if self._worker is not None and self._worker.isRunning():
            self._status(
                "A job is already in flight: it cannot be interrupted, "
                "so the calibration was not started."
            )
            return
        if self._current_mode() != MODE_HYDRO:
            self._status(
                "Calibration optimizes the hydrology model only; "
                "switch the header's *Hydrology only* checkbox on."
            )
            return
        table: pd.DataFrame = self._objective_model._table
        if table.empty:
            self._status(
                "No objective values to calibrate against yet: run the "
                "model first."
            )
            return
        effective: frozenset[str] = (
            self._objective_terms
            if self._objective_terms is not None
            else self._selectable_objective_terms()
        )
        terms: list[str] = [
            f"{target}: {metric}"
            for metric in table.index
            for target in table.columns
            if f"{target}: {metric}" in effective
        ]
        if not terms:
            self._status(
                "No objectives are selected: check at least one cell of "
                "the objective table."
            )
            return
        bounds: dict[str, tuple[float, float]] = (
            self._param_panel.current_bounds()
        )
        num_hydro: int = (
            self._input_data.spec.model_class.get_num_hydro_parameters()
        )
        hydro_names: tuple[str, ...] = tuple(self._input_data.bounds)[
            :num_hydro
        ]
        if not any(
            bounds[name][1] > bounds[name][0] for name in hydro_names
        ):
            self._status(
                "No free hydrology parameters to calibrate: give at least "
                "one hydrology cell a range."
            )
            return
        settings: CalibrationSettings | None = ask_calibration_settings(self)
        if settings is None:
            return
        self._start_calibration(settings, terms)

    def _start_calibration(
        self,
        settings: CalibrationSettings,
        terms: list[str],
    ) -> None:
        """
        Start the calibration worker and show its progress.

        Args:
            settings: The optimizer options the user entered.
            terms: The objective terms to optimize, as
                ``"target: metric"`` labels.
        """
        x: np.ndarray = self._current_x()
        bounds: dict[str, tuple[float, float]] = (
            self._param_panel.current_bounds()
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
        self._calibrate_button.setEnabled(False)
        self._status("Calibrating the hydrology model…")

        worker: _CalibrationWorker = _CalibrationWorker(
            x,
            self._input_data,
            bounds,
            terms,
            settings,
            self._input_data.spec.key,
            self,
        )
        worker.progress_line.connect(self._on_calibration_progress_line)
        worker.progress_step.connect(self._on_progress_step)
        worker.finished_calibration.connect(self._on_finished_calibration)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        self._elapsed_timer.start()
        worker.start()

    def _on_calibration_progress_line(self, line: str) -> None:
        """Show the calibration's latest progress line in the status bar."""
        if line:
            self._status(f"Calibration in progress — {line}")

    def _on_finished_calibration(self, payload: object) -> None:
        """
        Apply a finished calibration job's outcome to the window.

        A failed job is reported in the status bar and leaves the
        figures on screen unchanged.  A successful job adopts the
        best full parameter vector into the panel, shows the best
        run's results, refreshes the figures and the table, and saves
        the run.
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

        if not isinstance(payload, _CalibrationOutcome):
            return
        outcome: _CalibrationOutcome = payload
        if outcome.model_key != self._input_data.spec.key:
            # The model changed mid-job: the outcome belongs to the
            # model the window no longer shows.  Drop it.
            self._update_calibrate_button()
            self._update_run_button()
            return
        if (
            outcome.error is not None
            or outcome.x_best is None
            or outcome.results is None
        ):
            self._status(f"Calibration failed: {outcome.error}")
            up_to_date: bool = self._results_are_up_to_date()
            self._save_button.setEnabled(up_to_date)
            self._run_button.setEnabled(not up_to_date)
            self._update_calibrate_button()
            return

        bounds: dict[str, tuple[float, float]] = (
            self._param_panel.current_bounds()
        )
        best_values: dict[str, float] = {
            key: float(value)
            for key, value in zip(bounds, outcome.x_best, strict=True)
        }
        self._param_panel.restore_state(best_values, bounds)
        self._results = outcome.results
        self._last_run_x = outcome.x_best
        self._last_run_mode = MODE_HYDRO
        self._refresh_from_results()
        saved: Path | None = self._on_save_run()
        saved_text: str = (
            f"the best run was saved to {saved}."
            if saved is not None
            else "the best run could not be saved."
        )
        if len(outcome.terms) == 1:
            best_text: str = (
                f"({outcome.terms[0]} = {outcome.best_value:.3g})"
            )
        else:
            best_text = (
                f"(sum of {len(outcome.terms)} objectives = "
                f"{outcome.best_value:.3g})"
            )
        self._status(
            f"Calibration finished in {outcome.elapsed_seconds:.1f} s "
            f"after {outcome.evaluations} evaluations: best {best_text}; "
            f"the best parameters are in the panel and {saved_text}"
        )
        up_to_date = self._results_are_up_to_date()
        self._save_button.setEnabled(up_to_date)
        self._run_button.setEnabled(not up_to_date)
        self._update_calibrate_button()

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
            model_key=self._input_data.spec.key,
            network_fingerprint=fingerprint_network(self._network),
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
            model_key=self._input_data.spec.key,
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
                model_key=self._input_data.spec.key,
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
        item: _HistoryItem = self._history_items[index]
        if item.model_key != self._input_data.spec.key:
            return None
        return item

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
        self._set_mode_tint(mode)
        self._last_run_mode = mode
        self._save_button.setEnabled(True)
        # The restored results correspond to the restored parameters,
        # so there is nothing for a manual run to do yet.
        self._update_run_button()
        self._refresh_from_results()

    # ==== Parameter panel ==== #
    def _set_mode_tint(self, mode: str) -> None:
        """
        Tint the window's background by run mode.

        The window itself takes the mode's background: the cool
        hydrology tint for a hydrology-only run, the warm
        reactive-transport tint otherwise.  The parameter-panel
        groups keep their own theme tints, so only the window
        changes with the mode.

        Args:
            mode: The run mode (`MODE_HYDRO` or `MODE_FULL`).
        """
        tint: str = HYDRO_THEME.light if mode == MODE_HYDRO else RT_MODE_TINT
        self.setStyleSheet(f"background-color: {tint};")

    def _on_mode_changed(self, _enabled: bool) -> None:
        """
        The header's *Hydrology only* checkbox was toggled.

        The parameter panel hides or shows its reactive-transport
        parts, the window takes the new mode's tint, the results on
        screen are stale in the new mode (the run button reflects
        that), the line search's selection is kept a subset of the
        mode's selectable terms (the checkboxes repaint), and with
        auto-run on a debounced re-run follows so the figures match
        the mode.
        """
        mode: str = self._current_mode()
        self._param_panel.set_hydro_only(mode == MODE_HYDRO)
        self._set_mode_tint(mode)
        self._update_run_button()
        self._update_calibrate_button()
        self._sync_objective_selection(self._selectable_objective_terms())
        if self._auto_run_checkbox.isChecked():
            self._debounce_timer.start()

    def _on_model_changed(self, _index: int) -> None:
        """
        The header's model selector changed: switch to its model.

        The combo's item data is the selected model's spec key; a
        change to the key already loaded is ignored (a programmatic
        resync of the combo emits the same signal).
        """
        key: object = self._model_combo.currentData()
        if not isinstance(key, str) or key == self._input_data.spec.key:
            return
        self._switch_model(key)

    def _switch_selection(self, spec: ModelSpec) -> PlotSelection:
        """
        The plot selection after a model switch.

        The species, mineral, streamflow toggle, and date window are
        network- or record-level and survive the switch; the zones
        are the new model's defaults, since the old model's zone
        names may not exist in the new one.

        Args:
            spec: The model spec being switched to.

        Returns:
            The selection to hand the rebuilt parameter panel.
        """
        old: PlotSelection = self._param_panel.selection()
        return PlotSelection(
            species=list(old.species),
            zones=default_plot_selection(spec).zones,
            mineral=old.mineral,
            streamflow=old.streamflow,
            window_start=old.window_start,
            window_end=old.window_end,
        )

    def _switch_model(self, key: str, *, start_run: bool = True) -> None:
        """
        Reload the window for the model spec ``key``.

        The new model's inputs replace the loaded ones, the
        parameter panel is rebuilt at its new defaults, and every
        model-specific piece of state is discarded: the results,
        the figures, the objective table, the gradient and line
        search pages, and the session history.  The saved runs
        listed in the history dropdown are filtered to the new
        model.  The title and status follow the model.

        A worker still in flight cannot be interrupted, so the run
        the switch asks for is deferred to `_on_worker_finished`
        (the run's own outcome is discarded by the bumped
        generation); otherwise the run starts now when auto-run is
        on, or the run button lights up.

        Args:
            key: The spec key to switch to.
            start_run: Whether the switch may start a run.  Startup
                passes False: it applies a saved run immediately, so
                a fresh run would overwrite the restored results.
        """
        self._run_generation += 1
        self._debounce_timer.stop()
        self._start_x = None
        self._start_mode = None
        spec: ModelSpec = get_model_spec(key)
        self._input_data = load_input_data(
            Path(self._data_path) if self._data_path else DEFAULT_CSV_PATH,
            model_key=key,
            network=self._network,
            column_map=self._column_map,
        )
        self._param_panel.reload(
            self._input_data, self._switch_selection(spec)
        )
        self._results = None
        self._last_run_x = None
        self._last_run_mode = None
        self._gradient_state = None
        self._line_search_state = None
        self._figure_panel.set_figures([])
        self._objective_model.set_table(pd.DataFrame())
        self._fit_objective_panel()
        self._saved_run_cache.clear()
        self._history_items = [
            _HistoryItem(
                label=run.name,
                run_dir=run,
                param_values=None,
                results=None,
                mode=None,
                model_key=key,
            )
            for run in list_run_dirs()
            if saved_run_model_key(run) == key
            and _run_matches_network(run, fingerprint_network(self._network))
        ]
        self._sync_history_combo()
        self._title_label.setText(_title_for(spec))
        # The header can clip the title under squeeze; the tooltip
        # (set again here, since the text changed) keeps the full
        # title readable.
        self._title_label.setToolTip(_title_for(spec))
        self.setWindowTitle(_title_for(spec))
        self._status(f"Switched to {spec.display_name}.")
        if not start_run:
            return
        if self._worker is not None and self._worker.isRunning():
            self._pending_run_after_worker = True
            return
        if self._auto_run_checkbox.isChecked():
            self._start_run(self._current_x())
        else:
            self._run_button.setEnabled(True)
            self._gradients_button.setEnabled(True)
            self._line_search_button.setEnabled(True)

    def _sync_network_selector(self, name: str | None = None) -> None:
        """
        Sync the network selector to the network the window runs.

        A network matching a preset's fingerprint selects that
        preset's entry and drops any `custom:` entry; a network
        that matches no preset gets, or updates, the selector's
        single `custom: <name>` entry, keyed by the fingerprint so
        a re-sync of the same network never adds a second one.
        The signal is blocked for the whole update, since the
        switch methods call this on purpose, not on user input.

        Args:
            name: The display name of a custom network; the
                window's current custom name when None, and
                "edited" when there is none.  Ignored for presets.
        """
        fingerprint: str = fingerprint_network(self._network)
        preset_key: str | None = _preset_fingerprint_map().get(
            fingerprint
        )
        self._network_combo.blockSignals(True)
        try:
            if preset_key is not None:
                self._remove_custom_entry()
                self._custom_network_name = None
                self._network_combo.setCurrentIndex(
                    self._network_combo.findData(preset_key)
                )
            else:
                custom_name: str = (
                    name
                    or self._custom_network_name
                    or "edited"
                )
                self._custom_network_name = custom_name
                row: int = -1
                for index in range(self._network_combo.count()):
                    if self._network_combo.itemData(index) == fingerprint:
                        row = index
                if row >= 0:
                    self._network_combo.setItemText(
                        row, f"custom: {custom_name}"
                    )
                else:
                    self._remove_custom_entry()
                    self._network_combo.addItem(
                        f"custom: {custom_name}", fingerprint
                    )
                    row = self._network_combo.count() - 1
                self._network_combo.setCurrentIndex(row)
        finally:
            self._network_combo.blockSignals(False)

    def _remove_custom_entry(self) -> None:
        """Remove the selector's `custom:` entry, if it has one."""
        for index in range(self._network_combo.count()):
            if self._network_combo.itemData(index) not in NETWORK_SPECS:
                self._network_combo.removeItem(index)
                return

    def _on_network_changed(self, _index: int) -> None:
        """
        The network selector changed: switch to the selected
        network.

        A change to the fingerprint already running is ignored (a
        programmatic resync of the combo emits the same signal),
        and a `custom:` entry never switches here: it is only ever
        selected by `_sync_network_selector`, which already
        performed the switch.
        """
        data: object = self._network_combo.currentData()
        if not isinstance(data, str) or data not in NETWORK_SPECS:
            return
        if (
            _preset_fingerprint_map().get(
                fingerprint_network(self._network)
            )
            == data
        ):
            return
        self._switch_network(get_network_spec(data).builder())

    def _switch_network(
        self,
        network: pt.ReactionNetwork,
        *,
        name: str | None = None,
        start_run: bool = True,
    ) -> None:
        """
        Reload the window for the reaction network ``network``.

        The new network's inputs replace the loaded ones (the
        model spec is unchanged), the parameter panel's plot
        selection follows the network's measured species and
        dominant mineral, and every run-specific piece of state
        is discarded: the results, the figures, the objective
        table, the gradient and line search pages, and the
        session history.  The saved runs listed in the history
        dropdown are filtered to runs of the same model on this
        network's fingerprint (`_run_matches_network`: a run
        without a recorded fingerprint is treated as a run of the
        default network).  The network selector follows the
        switch.

        A worker still in flight cannot be interrupted, so the run
        the switch asks for is deferred to `_on_worker_finished`
        (the run's own outcome is discarded by the bumped
        generation); otherwise the run starts now when auto-run is
        on, or the run button lights up.

        Args:
            network: The network to switch to.
            name: The display name of a custom (non-preset)
                network; the window's current custom name when
                None, and "edited" when there is none.  Presets
                ignore the name: their entry is the preset's own.
            start_run: Whether the switch may start a run.  Startup
                passes False: it applies a saved run immediately, so
                a fresh run would overwrite the restored results.
        """
        self._run_generation += 1
        self._debounce_timer.stop()
        self._start_x = None
        self._start_mode = None
        self._network = network
        spec: ModelSpec = self._input_data.spec
        self._input_data = load_input_data(
            Path(self._data_path) if self._data_path else DEFAULT_CSV_PATH,
            model_key=spec.key,
            network=network,
            column_map=self._column_map,
        )
        new_default: PlotSelection = default_plot_selection(
            spec,
            network=network,
            window=self._input_data.window,
        )
        old: PlotSelection = self._param_panel.selection()
        self._param_panel.reload(
            self._input_data,
            PlotSelection(
                species=new_default.species,
                zones=old.zones,
                mineral=new_default.mineral,
                streamflow=old.streamflow,
                window_start=old.window_start,
                window_end=old.window_end,
            ),
        )
        self._results = None
        self._last_run_x = None
        self._last_run_mode = None
        self._gradient_state = None
        self._line_search_state = None
        self._figure_panel.set_figures([])
        self._objective_model.set_table(pd.DataFrame())
        self._fit_objective_panel()
        self._saved_run_cache.clear()
        self._history_items = [
            _HistoryItem(
                label=run.name,
                run_dir=run,
                param_values=None,
                results=None,
                mode=None,
                model_key=spec.key,
            )
            for run in list_run_dirs()
            if saved_run_model_key(run) == spec.key
            and _run_matches_network(run, fingerprint_network(network))
        ]
        self._sync_history_combo()
        self._sync_network_selector(name)
        self._status(
            f"Switched to network {self._network_combo.currentText()}."
        )
        if not start_run:
            return
        if self._worker is not None and self._worker.isRunning():
            self._pending_run_after_worker = True
            return
        if self._auto_run_checkbox.isChecked():
            self._start_run(self._current_x())
        else:
            self._run_button.setEnabled(True)
            self._gradients_button.setEnabled(True)
            self._line_search_button.setEnabled(True)

    def _switch_data(
        self,
        input_data: GuiInputData,
        data_path: str | None,
        column_map: dict[str, str | None] | None,
    ) -> None:
        """
        Reload the window for the record ``input_data``.

        The record's path and column mapping are stored on the
        window so later switches (model, network) and saves carry
        them, and the parameter panel is rebuilt with the
        record's forcing span as the default plot window.  Every
        run-specific piece of state is discarded, exactly as in
        `_switch_model`, with one difference: switching to a new
        record starts a new session, so the saved runs in the
        history dropdown are cleared rather than re-listed (the
        disk listing would mix runs of other records).

        The run (and the gradient and line search) buttons light
        up; a run the switch asks for is deferred to
        `_on_worker_finished` when a worker is still in flight,
        or started now when auto-run is on.
        """
        self._run_generation += 1
        self._debounce_timer.stop()
        self._start_x = None
        self._start_mode = None
        self._data_path = data_path
        self._column_map = column_map
        self._input_data = input_data
        self._param_panel.reload(
            self._input_data,
            default_plot_selection(
                self._input_data.spec,
                network=self._network,
                window=self._input_data.window,
            ),
        )
        self._results = None
        self._last_run_x = None
        self._last_run_mode = None
        self._gradient_state = None
        self._line_search_state = None
        self._figure_panel.set_figures([])
        self._objective_model.set_table(pd.DataFrame())
        self._fit_objective_panel()
        self._saved_run_cache.clear()
        self._history_items = []
        self._sync_history_combo()
        if data_path is None:
            self._status("Loaded the record.")
        else:
            self._status(f"Loaded {Path(data_path).name}.")
        if self._worker is not None and self._worker.isRunning():
            self._pending_run_after_worker = True
            return
        if self._auto_run_checkbox.isChecked():
            self._start_run(self._current_x())
        else:
            self._run_button.setEnabled(True)
            self._gradients_button.setEnabled(True)
            self._line_search_button.setEnabled(True)

    def _on_load_data(self) -> None:
        """
        Load a CSV or XLSX record and switch the window to it.

        The file dialog picks the record file.  `prefill_mapping`
        reads its columns and auto-detected role mapping; a file
        it cannot map — an unsupported suffix, an unreadable file,
        or no date column — is reported and the session keeps
        running.  The mapping dialog then confirms the mapping;
        accepting it loads the record through `load_input_data`
        with the mapping (a failure there — the mapping pointing
        at a missing column, an interior forcing gap — is
        reported the same way) and switches the window to the
        record.
        """
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Load data",
            "",
            "Data files (*.csv *.xlsx)",
        )
        if not path:
            return
        try:
            columns, prefill = prefill_mapping(Path(path), self._network)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Load data", str(error))
            return
        dialog: ColumnMappingDialog = ColumnMappingDialog(
            columns, self._network, prefill, self
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        mapping: dict[str, str | None] = dialog.mapping()
        try:
            input_data: GuiInputData = load_input_data(
                Path(path),
                model_key=self._input_data.spec.key,
                network=self._network,
                column_map=mapping,
            )
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Load data", str(error))
            return
        self._switch_data(input_data, path, mapping)

    def _on_load_network(self) -> None:
        """
        Load the network of a YAML file and run it.

        A file whose network matches a preset (same fingerprint)
        selects that preset's entry; any other network becomes the
        selector's `custom:` entry, named after the file's stem.
        A file that fails to load or validate is reported and
        ignored: the window keeps running its current network.
        """
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Load network file",
            "",
            "YAML files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            network: pt.ReactionNetwork = pt.ReactionNetwork.from_file(path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Load network", str(error))
            return
        self._switch_network(network, name=Path(path).stem)

    def _on_save_network(self) -> None:
        """
        Write the network currently running to a YAML file.

        The window keeps running the very same network: the write
        is a snapshot, and the selector is untouched.
        """
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save network as",
            "",
            "YAML files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            pt.NetworkFile.from_network(self._network).to_file(path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Save network", str(error))
        else:
            QMessageBox.information(
                self, "Save network", f"Network written to {path}."
            )

    def _on_edit_network(self) -> None:
        """
        Edit the running network in the modal dialog.

        Applying the dialog switches the window to the edited
        network: a result that matches a preset keeps the
        preset's entry, an edit of a preset gets the selector's
        `custom: edited` entry, and an edit of a custom network
        keeps its name.  Cancel or close leaves the window
        untouched.
        """
        database: pt.ChemicalDatabase = pt.ChemicalDatabase.load_default()
        edited: pt.ReactionNetwork | None = edit_network(
            self._network, database, self
        )
        if edited is None:
            return
        self._switch_network(edited)

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
    def _on_save_run(self) -> Path | None:
        """
        Save the current run: the existing per-run model output,
        the running network as ``network.yaml``, the figures as
        plotted, and a ``gui_state.json`` capturing the session
        state, including the network's fingerprint (the key the
        history filtering matches on).

        The run is saved only when the results on screen correspond
        to the current parameters and mode (a run still catching up
        blocks the save, since its output would be stale).  A
        full-mode run saves the ``hydro`` and ``rt`` bundle pair
        and a hydrology-only run the flat hydrology bundle; the
        state file records the run mode and the checked objective
        cells (restored, then intersected with the restored
        table's selectable cells).  Every figure currently
        loaded in the figure panel (the mode's own figures, and the
        gradient and line search pages when those jobs have
        finished) is saved as a PNG into a ``figures`` subfolder of
        the run directory.  The new run directory becomes the
        first entry in the panel's *History* group (the newest
        save), so it can be restored from the dropdown without a
        re-run.

        Returns:
            The new run directory, or None when there was nothing
            current to save.
        """
        if self._results is None:
            self._status("Nothing to save yet: the model has not run.")
            return None
        if not self._results_are_up_to_date():
            self._status(
                "The model is still catching up; let the current run "
                "finish before saving."
            )
            return None

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
            model_class: type[pt.Model] = self._input_data.spec.model_class
            num_hydro: int = model_class.get_num_hydro_parameters()
            hydro_model: HydrologicalModel = model_class.hydro_from_array(
                self._current_x()[:num_hydro], natural_scales=True
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

        # The network the run used, in the canonical file format;
        # a restore of this run reads it back (``_run_network``).
        # A failure here would leave a partially saved run (the
        # run saver has already written the directory) that the
        # restore path would misclassify as pre-fingerprint, so
        # it is reported and the save is abandoned.
        try:
            pt.NetworkFile.from_network(self._network).to_file(
                str(run_dir / NETWORK_FILENAME)
            )
        except (OSError, ValueError) as error:
            QMessageBox.critical(
                self,
                "Save run",
                f"Could not write the run's network file "
                f"({run_dir / NETWORK_FILENAME}): {error}",
            )
            return None

        # The client geometry (rather than the frame's): the restore
        # path applies these values with setGeometry, so saving the
        # frame size would grow the window on every save and restore.
        # While maximized the client rect *is* the screen area, so
        # the tracked normal (pre-maximize) geometry is saved with
        # the maximized flag instead; the restore then reopens the
        # window maximized from a sane normal geometry.
        maximized: bool = self.isMaximized()
        geom: QRect = (
            self._normal_geometry
            if maximized and self._normal_geometry is not None
            else self.geometry()
        )
        state: GuiState = GuiState(
            csv_path=self._data_path,
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
            objective_terms=(
                list(self._objective_terms)
                if self._objective_terms is not None
                else None
            ),
            model_key=self._input_data.spec.key,
            network_fingerprint=fingerprint_network(self._network),
            column_map=self._column_map,
            window_maximized=maximized,
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
        return run_dir

    # ==== Refresh ==== #
    def _refresh_from_results(self) -> None:
        """
        Rebuild the figures and the objective table from the last
        results.

        Every figure honors the current plot selection — the river
        species, the zones, the mineral, the streamflow toggle, and
        the date window — so a selection change refreshes
        them without a re-run.  A hydrology-only run's results
        carry no reactive-transport simulation, so the three
        hydrology figures (hydrology, storage, partitioning) are
        shown and the objective table carries only the streamflow
        column; a full run adds the five reactive-transport
        figures (river concentrations, river loads, concentration
        vs. discharge, zone concentrations, and the selected
        mineral's reaction rates).  A previous run selected in the
        panel's *History* group (with its overlay checkbox on) is
        drawn faintly on every figure, behind the current run's
        series.  When a gradient job has finished, its difference
        matrix is drawn as an extra page after the mode's own
        figures, and when a line search has finished, its value
        profile rides along as a further page.
        """
        if self._results is None:
            return
        hydro: pt.HydroModelResults = self._results.hydro
        rt: pt.RtModelResults | None = self._results.rt
        selection: PlotSelection = self._param_panel.selection()

        # Every figure is clipped to the panel's date window, so a
        # range past the record's last day (the partial 2025 water
        # year) still plots only the days the record carries.
        window: tuple[pd.Timestamp, pd.Timestamp] = selection.plot_window()

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

        spec: ModelSpec = self._input_data.spec
        hydro_sim: pd.DataFrame = hydro.simulation
        hydrology: Figure = make_hydrology_figure(
            hydro_sim,
            self._input_data.q_meas,
            window,
            overlay_hydro_sim=overlay_hydro,
        )
        storage: Figure = make_storage_figure(
            hydro_sim,
            selection.zones,
            window,
            overlay_hydro_sim=overlay_hydro,
            zone_labels=spec.zone_labels,
            zone_colors=spec.zone_colors,
        )
        partitioning: Figure = make_partitioning_figure(
            hydro_sim,
            self._input_data.q_meas,
            window,
            overlay_hydro_sim=overlay_hydro,
            zone_labels=spec.zone_labels,
            zone_colors=spec.zone_colors,
            stack_order=spec.stack_order,
        )
        figures: tuple[tuple[str, Figure], ...] = (
            ("Hydrology", hydrology),
            ("Storage", storage),
            ("Partitioning", partitioning),
        )
        if rt is None:
            # A hydrology-only run shows the three hydrology
            # figures, with no reactive-transport column in the
            # table.
            table: pd.DataFrame = build_objective_table(
                hydro.objective_functions, None
            )
        else:
            rt_sim: pd.DataFrame = rt.simulation
            river_concentrations: Figure = (
                make_river_concentrations_figure(
                    rt_sim,
                    hydro_sim,
                    self._input_data.q_meas,
                    self._input_data.conc_df,
                    selection.species,
                    selection.streamflow,
                    window,
                    overlay_rt_sim=overlay_rt,
                    overlay_hydro_sim=overlay_hydro,
                )
            )
            river_loads: Figure = make_river_loads_figure(
                rt_sim,
                hydro_sim,
                selection.species,
                window,
                overlay_rt_sim=overlay_rt,
                overlay_hydro_sim=overlay_hydro,
            )
            conc_discharge: Figure = make_conc_discharge_figure(
                rt_sim,
                hydro_sim,
                self._input_data.q_meas,
                self._input_data.conc_df,
                selection.species,
                window,
            )
            zone_concentrations: Figure = make_zone_concentrations_figure(
                rt_sim,
                selection.species,
                selection.zones,
                window,
                overlay_rt_sim=overlay_rt,
                zone_labels=spec.zone_labels,
                zone_colors=spec.zone_colors,
            )
            rate: Figure = make_rate_figure(
                rt_sim,
                selection.mineral,
                window,
                overlay_rt_sim=overlay_rt,
                zone_labels=spec.zone_labels,
                zone_colors=spec.zone_colors,
                zone_order=spec.zones,
            )
            figures = figures + (
                ("River concentrations", river_concentrations),
                ("River loads", river_loads),
                ("Concentration vs. discharge", conc_discharge),
                ("Zone concentrations", zone_concentrations),
                (f"Reaction rates ({selection.mineral})", rate),
            )
            table = build_objective_table(
                hydro.objective_functions, rt.objective_functions
            )

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
            if len(search.terms) == 1:
                note = (
                    f"Steepest ascent of {search.terms[0]}, "
                    f"{_LINE_SEARCH_POINTS} points within the current "
                    f"bounds; saved to {search.csv_path}."
                )
            else:
                note = (
                    f"Steepest ascent of {len(search.terms)} "
                    f"equal-weighted objectives "
                    f"({', '.join(search.terms)}), "
                    f"{_LINE_SEARCH_POINTS} points within the current "
                    f"bounds; saved to {search.csv_path}."
                )
            mode_label = "hydro" if search.mode == MODE_HYDRO else "full"
            search_fig: Figure = make_line_search_figure(
                search.profile, search.terms, note
            )
            figures = figures + (
                (f"Line search ({mode_label})", search_fig),
            )

        self._figure_panel.set_figures(figures)
        self._objective_model.set_table(table)
        # The new table's selectable cells define what the line
        # search can search: keep the checked terms a subset of
        # them (a term that lost its finite value drops out) and
        # repaint the checkboxes.
        self._sync_objective_selection(self._selectable_objective_terms())
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

    # ==== Window-state events ==== #
    def changeEvent(self, event: QEvent) -> None:
        """
        Capture the normal geometry the instant the window maximizes.

        Qt delivers `WindowStateChange` before resizing the window to
        the maximized client area, so at that instant `geometry()` is
        still the normal geometry — the value a maximized *Save run*
        records.  Un-maximizing delivers the same event with the
        maximized geometry, which is deliberately ignored.
        """
        if event.type() == QEvent.Type.WindowStateChange and self.isMaximized():
            self._normal_geometry = self.geometry()
        super().changeEvent(event)

    def resizeEvent(self, event: QEvent) -> None:
        """Keep the normal geometry current while the window resizes normally."""
        if not self.isMaximized():
            self._normal_geometry = self.geometry()
        super().resizeEvent(event)

    def moveEvent(self, event: QEvent) -> None:
        """Keep the normal geometry current while the window moves normally."""
        if not self.isMaximized():
            self._normal_geometry = self.geometry()
        super().moveEvent(event)

    def showEvent(self, event: QEvent) -> None:
        """
        Apply a deferred restore-maximize when the window is shown.

        A restored run that was saved maximized maximizes here rather
        than through its saved client rect.  The maximize is deferred
        one turn through the event loop: maximizing synchronously
        inside `showEvent` makes Qt deliver the state changes out of
        order (a spurious un-maximized `WindowStateChange` after the
        maximized resize on the offscreen plugin), which would corrupt
        the tracked normal geometry.
        """
        if self._restore_maximized:
            self._restore_maximized = False
            QTimer.singleShot(0, self.showMaximized)
        super().showEvent(event)

    def _toggle_maximized(self) -> None:
        """Toggle the window between maximized and its normal geometry."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

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
