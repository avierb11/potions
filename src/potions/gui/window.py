"""
The main window of the East River lateral-model calibration GUI.

The window stacks three splitter sections: the figure panel (the
model's three figures, one visible at a time, flipped by the panel's
arrow buttons or the main window's Left / Right keys), an always
visible strip with the per-run objective-function table, and the
parameter panel below the figure.

Parameter changes re-run the models on a worker thread (a short
debounce covers a drag or a keyboard step) and refresh the figures
and the table when the run finishes; a generation counter discards
stale results if the parameters change mid-run.  Plot-selection
changes rebuild the zone and rate figures from the last results
without re-running.

At startup the window restores the most recent saved run: sliders,
edited bounds, plot selections, the window geometry, and the
splitter sizes come from the run's ``gui_state.json`` (or, for an
older run, best-effort from its parameters CSVs), and the figures
are built straight from the run's saved simulation output, so the
window opens without a re-run.  With no saved run it starts from
the default parameters and runs the models once in the background.

A *Save run* click writes the existing per-run model output plus a
``gui_state.json`` capturing the session state, so the next start
restores exactly what the user saw.
"""

import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import potions as pt
from matplotlib.figure import Figure
from potions.result_saving import save_model_run
from PySide6.QtCore import (
    QAbstractTableModel,
    QCoreApplication,
    QObject,
    QModelIndex,
    QPersistentModelIndex,
    QThread,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QCloseEvent, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    PlotSelection,
    arr_to_model,
    arr_to_results,
    default_plot_selection,
)
from potions.gui.figure_panel import FigurePanel
from potions.gui.param_panel import ParamPanel
from potions.gui.plots import (
    build_objective_table,
    make_rate_figure,
    make_time_series_figure,
    make_zone_figure,
)
from potions.gui.state import (
    GuiState,
    SavedRun,
    find_latest_run_dir,
    load_saved_run,
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
        results: The model results, or None when the run failed.
        error: A one-line description of the failure, or None.
        elapsed_seconds: The wall-clock duration of the run.
    """

    generation: int
    results: pt.ModelResults | None
    error: str | None
    elapsed_seconds: float


# ==== Worker ==== #
class _RunWorker(QThread):
    """
    A worker thread running the hydrology and reactive-transport
    models off the event loop.

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
        parent: The parent object (the window); defaults to None.
    """

    finished_run = Signal(object)
    progress_line = Signal(str)

    def __init__(
        self,
        generation: int,
        x: np.ndarray,
        input_data: GuiInputData,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._generation: int = generation
        self._x: np.ndarray = x
        self._input_data: GuiInputData = input_data

    def run(self) -> None:
        """Run the models and emit the outcome, capturing stdout lines."""
        # The Any annotation is the type of sys.stdout in the
        # stdlib stubs; the value is a real text stream at runtime.
        real_stdout: Any = sys.stdout
        sys.stdout = _CapturingStream(self.progress_line.emit, real_stdout)
        started: float = time.perf_counter()
        results: pt.ModelResults | None = None
        error: str | None = None
        try:
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


# ==== Objective table ==== #
class _ObjectiveTableModel(QAbstractTableModel):
    """
    A read-only item model over the per-run objective-function
    table, for the table strip under the figure.

    Cells render their values at four decimals (the resolution of
    the saved objective CSVs) with a placeholder for a missing or
    NaN value; the row and column headers are the table's own
    (species labels and metric labels).

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
        """The number of species rows (zero for any child parent)."""
        return 0 if parent.isValid() else len(self._table)

    def columnCount(
        self, parent: QModelIndex | QPersistentModelIndex = _INVALID_INDEX
    ) -> int:
        """The number of metric columns (zero for any child parent)."""
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
        """The display text of a cell: the value at four decimals, or a placeholder."""
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
        return f"{float(cell):.4f}"

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = 0
    ) -> str | None:
        """The row (species) or column (metric) label of a header section."""
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._table.columns[section])
        return str(self._table.index[section])


# ==== Main window ==== #
class MainWindow(QMainWindow):
    """
    The main window of the East River lateral-model calibration GUI.

    The window stacks three non-collapsible splitter sections: the
    figure panel (the model's three figures, one visible at a time),
    an always visible strip with the per-run objective-function
    table, and the parameter panel.  The header row carries the
    title, a status line, an indeterminate progress bar and
    elapsed-time label (visible only while a run is in flight), and
    the *Reset bounds* and *Save run* buttons.

    Parameter changes re-run the models on a worker thread, 400 ms
    after the last change, and refresh the figures and the table
    when the run finishes; a generation counter discards stale
    results and starts the next run with the latest values.  A run
    that fails (for instance with ``WaterVolumeError``) is
    reported in the status bar and the last valid figures stay on
    screen.  *Save run* is disabled while a run is in flight.

    At startup the window restores the most recent saved run (its
    parameters, edited bounds, plot selections, window geometry,
    and splitter sizes) and builds the figures from the run's
    saved output, so no re-run is needed; with no saved run it
    starts from the default parameters and runs once in the
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
        self._results: pt.ModelResults | None = None
        # The run currently in flight (None while no run is in
        # flight; the reference is dropped when the worker ends,
        # before the deferred deletion removes the object).
        self._worker: _RunWorker | None = None
        # The generation of the latest started run; an outcome with
        # an older generation is stale and discarded.
        self._run_generation: int = 0
        # The parameter vector of the latest started run (None
        # while no run has started).
        self._start_x: np.ndarray | None = None
        # The parameter vector of the latest finished run (None
        # until one has finished): the results on screen correspond
        # to exactly this vector.
        self._last_run_x: np.ndarray | None = None
        # The start time of the run in flight, for the elapsed
        # label; None when no run is in flight.
        self._run_started_at: float | None = None

        self.setWindowTitle(_TITLE)
        self.resize(*_INITIAL_SIZE)

        self._build_ui()
        self._build_actions()
        self._restore_or_start()

    # ==== Building ==== #
    def _build_ui(self) -> None:
        """Assemble the header row and the three splitter sections."""
        header: QWidget = self._build_header()

        self._figure_panel: FigurePanel = self._build_figure_panel()
        objective_panel: QWidget = self._build_objective_panel()
        param_panel: ParamPanel = self._build_param_panel()

        self._splitter: QSplitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self._figure_panel)
        self._splitter.addWidget(objective_panel)
        self._splitter.addWidget(param_panel)
        for section in range(self._splitter.count()):
            self._splitter.setCollapsible(section, False)

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

        self._status_label: QLabel = QLabel("Preparing…")
        self._status_label.setStyleSheet("color: gray;")

        self._progress_bar: QProgressBar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedWidth(_PROGRESS_WIDTH)
        self._progress_bar.hide()

        self._elapsed_label: QLabel = QLabel("")
        self._elapsed_label.hide()

        self._reset_button: QPushButton = QPushButton("Reset bounds")
        self._reset_button.clicked.connect(self._on_reset_bounds)

        self._save_button: QPushButton = QPushButton("Save run")
        self._save_button.clicked.connect(self._on_save_run)
        self._save_button.setEnabled(False)

        layout: QHBoxLayout = QHBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self._status_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._elapsed_label)
        layout.addWidget(self._reset_button)
        layout.addWidget(self._save_button)
        header: QWidget = QWidget(self)
        header.setLayout(layout)
        return header

    def _build_figure_panel(self) -> FigurePanel:
        """Build the figure panel (its header carries the arrow buttons)."""
        return FigurePanel(self)

    def _build_objective_panel(self) -> QWidget:
        """Build the strip under the figure: heading, table, and note."""
        self._objective_model: _ObjectiveTableModel = _ObjectiveTableModel(self)
        table: QTableView = QTableView(self)
        table.setModel(self._objective_model)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setShowGrid(False)
        table.verticalHeader().setVisible(False)
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

        With a restorable latest run the window opens on that run's
        figures and parameters without a re-run; otherwise (no run,
        or an unreadable one) it starts from the default parameters
        and runs the models once in the background.
        """
        latest: Path | None = find_latest_run_dir()
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

        The panel takes the saved values and bounds (a degenerate
        saved bound pins its parameter), the checkbox selection and
        the window geometry and splitter sizes are restored, and the
        figures and the table are built from the run's saved
        results.  The saved parameter vector is marked as the last
        run, so *Save run* is immediately available.
        """
        self._param_panel.restore_state(saved.param_values, saved.param_bounds)
        self._param_panel.set_selection(saved.selection)
        self._results = saved.results
        self._last_run_x = self._current_x()

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
            self._splitter.setSizes(saved.splitter_sizes)
        self._save_button.setEnabled(True)
        self._refresh_from_results()

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

    def _maybe_start_run(self) -> None:
        """
        Start a run with the current parameters, unless there is
        nothing to do.

        Skipped while a run is in flight, and when the results on
        screen already correspond to the current parameters (no
        change since the last finished run).
        """
        if self._worker is not None and self._worker.isRunning():
            return
        x: np.ndarray = self._current_x()
        if (
            self._last_run_x is not None
            and self._results is not None
            and np.array_equal(x, self._last_run_x)
        ):
            return
        self._start_run(x)

    def _start_run(self, x: np.ndarray) -> None:
        """Start a worker run with `x` and show the run indicators."""
        self._run_generation += 1
        self._start_x = x
        self._run_started_at = time.perf_counter()
        self._progress_bar.show()
        self._elapsed_label.show()
        self._save_button.setEnabled(False)
        self._status("Running the model…")

        worker: _RunWorker = _RunWorker(
            self._run_generation, x, self._input_data, self
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
        started run is stale (the parameters changed mid-run) and is
        discarded; the window then starts a fresh run with the
        latest values.  A failed run is reported in the status bar
        and the last valid figures stay on screen.
        """
        self._elapsed_timer.stop()
        self._progress_bar.hide()
        self._elapsed_label.hide()

        if not isinstance(payload, _RunOutcome):
            return
        outcome: _RunOutcome = payload
        if outcome.generation != self._run_generation or self._start_x is None:
            return
        completed_x: np.ndarray = self._start_x
        self._start_x = None
        self._run_started_at = None

        if outcome.error is not None:
            self._status(f"Run failed: {outcome.error}")
        elif outcome.results is None:
            self._status("Run failed: the worker reported no results.")
        else:
            self._results = outcome.results
            self._last_run_x = completed_x
            self._status(
                f"Run {self._run_generation} finished in "
                f"{outcome.elapsed_seconds:.1f} s."
            )
            self._refresh_from_results()

        up_to_date: bool = (
            self._last_run_x is not None
            and np.array_equal(completed_x, self._last_run_x)
        )
        self._save_button.setEnabled(up_to_date)
        # The parameters changed while this run was in flight: the
        # results are stale, so start the next one with the latest
        # values (this is what the generation counter is for).
        if up_to_date and not np.array_equal(completed_x, self._current_x()):
            self._start_run(self._current_x())

    # ==== Parameter panel ==== #
    def _on_value_changed(self, _key: str) -> None:
        """A parameter value changed: re-run after the debounce."""
        self._debounce_timer.start()

    def _on_bounds_changed(self, _key: str, _lo: float, _hi: float) -> None:
        """A parameter range changed (the value changed with it)."""
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
        self._debounce_timer.start()

    # ==== Save ==== #
    def _on_save_run(self) -> None:
        """
        Save the current run: the existing per-run model output plus
        a ``gui_state.json`` capturing the session state.

        The run is saved only when the results on screen correspond
        to the current parameters (a run still catching up blocks
        the save, since its output would be stale).
        """
        if self._results is None:
            self._status("Nothing to save yet: the model has not run.")
            return
        if self._last_run_x is None or not np.array_equal(
            self._current_x(), self._last_run_x
        ):
            self._status(
                "The model is still catching up; let the current run "
                "finish before saving."
            )
            return

        # ModelResults keeps no reference to the model that produced
        # it, and the run saver needs one (for the saved parameter
        # table and the run metadata), so the model is rebuilt from
        # the same parameter array: arr_to_model is construction
        # only, with the same defaults as the arr_to_results call
        # above, so it is the exact model that produced this run.
        model: pt.Model
        _initial_state: np.ndarray
        model, _initial_state = arr_to_model(self._current_x(), self._input_data)
        run_dir: Path = save_model_run(
            model,
            self._results.hydro,
            self._results.reactive_transport,
            results_dir=None,
        )

        state: GuiState = GuiState(
            csv_path=self._csv_path,
            param_values=self._param_panel.param_values(),
            param_bounds=self._param_panel.current_bounds(),
            selection=self._param_panel.selection(),
            window_geometry={
                "x": self.x(),
                "y": self.y(),
                "width": self.frameGeometry().width(),
                "height": self.frameGeometry().height(),
            },
            splitter_sizes=[int(size) for size in self._splitter.sizes()],
        )
        write_gui_state(run_dir, state)
        self._status(f"Saved run to {run_dir}.")

    # ==== Refresh ==== #
    def _refresh_from_results(self) -> None:
        """
        Rebuild the figures and the objective table from the last
        results.

        The zone and rate figures honor the current plot selection,
        so a selection change refreshes them without a re-run.
        """
        if self._results is None:
            return
        hydro: pt.HydroModelResults = self._results.hydro
        rt: pt.RtModelResults = self._results.reactive_transport
        selection: PlotSelection = self._param_panel.selection()

        hydro_sim: pd.DataFrame = hydro.simulation
        rt_sim: pd.DataFrame = rt.simulation

        time_series: Figure = make_time_series_figure(
            hydro_sim, rt_sim, self._input_data.q_meas, self._input_data.conc_df
        )
        zone: Figure = make_zone_figure(
            rt_sim, hydro_sim, selection.species, selection.zones
        )
        rate: Figure = make_rate_figure(rt_sim, selection.minerals)

        self._figure_panel.set_figures(
            (
                ("Streamflow and river concentrations", time_series),
                ("Zone concentrations and storage", zone),
                ("Mineral reaction rates", rate),
            )
        )
        table: pd.DataFrame = build_objective_table(
            hydro.objective_functions, rt.objective_functions
        ).round(4)
        self._objective_model.set_table(table)

    # ==== Status ==== #
    def _status(self, text: str) -> None:
        """Set the status line of the header row."""
        self._status_label.setText(text)

    # ==== Closing ==== #
    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Close the window, waiting for a run in flight to finish.

        Runs cannot be interrupted: when a run is in progress the
        user is asked whether to wait, and the event loop pumps
        until the worker finishes (the window stays responsive and
        shows the final results) or the close is cancelled.
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
