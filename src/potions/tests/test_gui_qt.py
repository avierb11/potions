"""
Qt (PySide6) tests for the East River lateral-model calibration GUI.

These are the interactive-layer counterparts to `test_gui`, which
covers the data layer and the pure functions.  Here the actual
widgets are driven offscreen: the main window (startup, restoring a
saved run, background runs, saving, the gradients and line search
jobs, the history dropdown's restore and overlay, and closing with
a run in flight), the figure panel, the objective table model, and
the parameter panel (spin-box cells, pinned cells, their dialogs,
and the plot selection).

The offscreen Qt platform is requested before the first PySide6
import, and the model itself is never run: the window's
`arr_to_results` and `arr_to_hydro_results` calls are patched
with fakes that return the synthetic results from `test_gui`
(a real run takes about five minutes and is covered by that
file's own slow test).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QAction, QCloseEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayoutItem,
    QMenu,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import potions as pt
from potions.hydro_model import HydrologicalModel
from potions.reactive_transport.rt_zone import WaterVolumeError

from ..gui import data as gui_data
from ..gui import figure_panel as gui_figure_panel
from ..gui import param_panel as gui_param_panel
from ..gui import window as gui_window
from ..gui.common import (
    HYDRO_THEME,
    METRIC_LABELS,
    METRIC_NAMES,
    RT_THEME,
    SPECIATION_ZONES,
    TIME_SERIES_OPTIONS,
    param_label,
)
from ..gui.data import PlotSelection
from ..gui.state import (
    MODE_FULL,
    MODE_HYDRO,
    RUN_HISTORY_FILENAME,
    read_gui_state,
)
from .test_gui import (
    _alpha,
    _full_state,
    _lines,
    _midpoint_vector,
    _synthetic_results,
    _write_saved_run,
    input_data,
)


# ------------------------------------------------------------------ #
# Fixtures and helpers
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """A single offscreen QApplication shared by every test."""
    # `QApplication.instance()` is typed in the stubs to return the
    # QCoreApplication base class, hence the isinstance check.
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    assert isinstance(instance, QApplication)
    return instance


def _wait_until(predicate: Callable[[], bool], timeout: float = 15.0) -> None:
    """
    Pump the Qt event loop until `predicate` holds.

    Args:
        predicate: The condition to wait for.
        timeout: Seconds to wait before failing the test.
    """
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    deadline: float = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail(f"Timed out after {timeout} s: {predicate}")


def _pump(iterations: int = 50) -> None:
    """Run a few event-loop ticks (lets deferred deletions land)."""
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    for _ in range(iterations):
        app.processEvents()
        time.sleep(0.002)


def _fake_results_factory(
    calls: list[np.ndarray],
    input_data: gui_data.GuiInputData,
    hydro_only: bool = False,
) -> Callable[
    [np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults
]:
    """
    A stand-in for `arr_to_results` (or `arr_to_hydro_results`)
    that records each parameter vector it is called with and
    returns the synthetic results wrapped in a `GuiRunResults`.
    With `hydro_only` the reactive-transport result is None, as in
    a hydrology-only run.
    """

    def fake(
        x: np.ndarray, _input_data: gui_data.GuiInputData
    ) -> gui_data.GuiRunResults:
        calls.append(np.array(x))
        hydro, rt = _synthetic_results(input_data)
        if hydro_only:
            return gui_data.GuiRunResults(hydro, None)
        return gui_data.GuiRunResults(hydro, rt)

    return fake


def _make_window(
    input_data: gui_data.GuiInputData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: Callable[[np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults],
) -> gui_window.MainWindow:
    """
    Build a main window in an empty `tmp_path` (no saved run to
    restore) with `fake` patched in for `arr_to_results`.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gui_window, "arr_to_results", fake)
    return gui_window.MainWindow(input_data)


def _close_window(win: gui_window.MainWindow) -> None:
    """Stop the window's timers and drop it."""
    win._debounce_timer.stop()
    win.deleteLater()
    _pump()


def _assert_restored_dir(win: gui_window.MainWindow, expected: Path) -> None:
    """
    Assert that `win` holds results restored from `expected`.

    The API's `saved_dir` field is optional, so missing results or a
    missing directory fail the assertion: both call sites restore a
    run that must have recorded its directory.
    """
    results: gui_data.GuiRunResults | None = win._results
    assert results is not None, "Expected restored results to be set."
    saved_dir: str | None = results.hydro.saved_dir
    assert saved_dir is not None, "Expected the results to record a saved_dir."
    assert Path(saved_dir).resolve() == expected


def _install_bounds_stub(
    monkeypatch: pytest.MonkeyPatch,
    answer: tuple[float, float] | None,
) -> list[dict[str, object]]:
    """
    Replace `param_panel._BoundsDialog` with a recording stub.

    Each constructed dialog is recorded (title, prefilled ends,
    hint, and the value the static option would pin) and `run()`
    returns `answer` (None stands in for a cancelled dialog).

    Returns:
        The list of recorded dialog constructions.
    """
    records: list[dict[str, object]] = []

    class _StubDialog:
        def __init__(
            self,
            title: str,
            lo: float,
            hi: float,
            hint: str | None,
            value: float,
            parent: object | None = None,
        ) -> None:
            records.append(
                {
                    "title": title,
                    "lo": lo,
                    "hi": hi,
                    "hint": hint,
                    "value": value,
                }
            )

        def run(self) -> tuple[float, float] | None:
            return answer

    monkeypatch.setattr(gui_param_panel, "_BoundsDialog", _StubDialog)
    return records


def _double_click_event() -> QMouseEvent:
    """A plain left-button double-click mouse event."""
    return QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        QPointF(10.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _band_of(cell: QWidget) -> str:
    """
    The row band a grid cell's stylesheet asks for (its
    background-color).
    """
    match = re.search(r"background-color: (\S+);", cell.styleSheet())
    assert match is not None, f"No band in {cell.styleSheet()!r}"
    return match.group(1)


class _FakeMessageBox:
    """
    Stands in for `QMessageBox`; the window only calls `question`
    (which returns the class-level `answer`) and `critical`
    (whose calls are recorded, for the water-volume failure
    dialog).
    """

    class StandardButton:
        No: int = 0
        Yes: int = 1

    answer: int = StandardButton.No
    critical_calls: list[tuple[str, str]] = []

    @staticmethod
    def question(
        parent: object | None,
        title: str,
        text: str,
        buttons: int = 0,
    ) -> int:
        return _FakeMessageBox.answer

    @staticmethod
    def critical(
        parent: object | None,
        title: str,
        text: str,
        *buttons: object,
    ) -> None:
        _FakeMessageBox.critical_calls.append((title, text))


# ------------------------------------------------------------------ #
# MainWindow: startup
# ------------------------------------------------------------------ #


class TestMainWindowStartup:
    """Starting the window: from defaults, or restoring a saved run."""

    def test_starts_from_defaults_and_runs_once(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no saved run, run the models once at the defaults."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump(50)
            # Exactly one run was started, at the default (midpoint)
            # vector, and the results are on screen.
            assert len(calls) == 1
            assert np.array_equal(calls[0], _midpoint_vector(input_data))
            # The finished worker's reference was dropped and its
            # deferred deletion has landed: a later "maybe start"
            # must neither touch the deleted object nor re-run.
            win._maybe_start_run()
            assert len(calls) == 1
            assert re.fullmatch(
                r"Run 1 finished in \d+\.\d s\.", win._status_label.text()
            )
            assert win._save_button.isEnabled()
            # The run-history toggle starts off.
            assert not win._save_history_checkbox.isChecked()
            # The results on screen correspond to the current
            # parameters, so a manual run would do nothing.
            assert not win._run_button.isEnabled()
            # The progress strip (bar and elapsed label) is hidden when
            # no job is in flight.  `isHidden()` only reflects a
            # widget's own visibility flag, so assert the strip rather
            # than its children.
            assert win._progress_strip.isHidden()
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            assert panel.page_count() == 3
            assert panel.current_page() == 0
            assert panel._titles == [
                "Streamflow and river concentrations",
                "Zone concentrations and storage",
                "Mineral reaction rates",
            ]
            # The table is transposed: one row per metric, one column
            # per measured target (the streamflow and the species).
            assert win._objective_model.rowCount() == 7
            assert win._objective_model.columnCount() == 4
            assert win.windowTitle() == gui_window._TITLE
        finally:
            _close_window(win)

    def test_objective_panel_fits_content_without_scrollbars(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The objective table needs no scrollbars and sits top-aligned."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump(50)
            # Both scrollbars are off: the view never needs to scroll.
            assert (
                win._objective_table.horizontalScrollBarPolicy()
                == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            assert (
                win._objective_table.verticalScrollBarPolicy()
                == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            # The panel is fixed in size (setFixedSize pins its
            # minimum to its maximum) to exactly what the table's
            # own measurements demand, per the pure sizing function.
            assert (
                win._objective_panel.minimumWidth()
                == win._objective_panel.maximumWidth()
                > 0
            )
            assert (
                win._objective_panel.minimumHeight()
                == win._objective_panel.maximumHeight()
                > 0
            )
            width, height = gui_window._objective_panel_size(
                win._objective_heading.sizeHint(),
                [
                    win._objective_table.horizontalHeader().sectionSizeHint(c)
                    for c in range(win._objective_model.columnCount())
                ],
                [
                    win._objective_table.verticalHeader().sectionSizeHint(r)
                    for r in range(win._objective_model.rowCount())
                ],
                win._objective_table.verticalHeader().sizeHint().width(),
                win._objective_table.horizontalHeader().sizeHint().height(),
                win._objective_table.frameWidth(),
                gui_window._OBJECTIVE_PANEL_MARGINS,
                gui_window._OBJECTIVE_PANEL_SPACING,
            )
            assert (
                win._objective_panel.width(),
                win._objective_panel.height(),
            ) == (width, height)
            # The zero-margin row layout top-aligns the panel.
            assert win._objective_panel.y() == 0
            # The heading is the panel's first child and is not
            # hidden.  `isVisible()` is false on an offscreen
            # window that was never `show()`-ed, so assert the
            # widget's own flag (as the progress-strip assertion
            # does).
            assert not win._objective_heading.isHidden()
        finally:
            _close_window(win)

    def test_restores_full_state(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A complete gui_state restores everything without a re-run."""
        run_dir: Path = _write_saved_run(
            tmp_path,
            input_data,
            _full_state(input_data, save_history=True),
        )
        calls: list[np.ndarray] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            gui_window,
            "arr_to_results",
            _fake_results_factory(calls, input_data),
        )
        win = gui_window.MainWindow(input_data)
        try:
            # No run was started: the saved results are used as is
            # (the saved directory is kept as the relative path the
            # lookup returned, so resolve it against the cwd).
            assert calls == []
            assert win._worker is None
            _assert_restored_dir(win, run_dir)
            assert win._status_label.text() == (
                "Restored the most recent saved run; the figures were "
                "built from its saved output."
            )
            assert win._save_button.isEnabled()
            # The restored results match the restored parameters.
            assert not win._run_button.isEnabled()
            panel: gui_param_panel.ParamPanel = win._param_panel
            assert panel.param_values()["proportion.1"] == pytest.approx(0.7)
            assert panel.current_bounds() == dict(input_data.bounds)
            assert panel.selection() == gui_data.default_plot_selection()
            # The save-history toggle is part of the saved state.
            assert win._save_history_checkbox.isChecked()
            assert win.x() == 10
            assert win.y() == 20
            assert win.width() == 1100
            assert win.height() == 800
            # Show the window first: an unshown window would normalize
            # the requested sizes against the window's size hint
            # instead of its restored geometry (where the figure
            # panel's minimum height would dominate the split).
            win.show()
            _pump()
            # Showing the window can shift it (the offscreen plugin
            # offsets it by its fake frame); re-assert the restored
            # size so the splitter lays out against the intended
            # geometry.
            win.resize(1100, 800)
            _pump()
            # The requested sizes are scaled to the splitter's
            # actual size, with each section clamped to its child's
            # minimum size hint: the proportions survive up to that
            # clamping, so a loose tolerance is the right assertion.
            sizes: list[float] = [float(s) for s in win._splitter.sizes()]
            total: float = sum(sizes)
            assert total > 0
            for size, target in zip(sizes, (300, 600)):
                assert size / total == pytest.approx(target / 900, abs=0.05)
            assert win._figure_panel.page_count() == 3
        finally:
            _close_window(win)

    def test_restores_legacy_run(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A run saved before gui_state exists restores from CSVs only."""
        run_dir: Path = _write_saved_run(tmp_path, input_data)
        calls: list[np.ndarray] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            gui_window,
            "arr_to_results",
            _fake_results_factory(calls, input_data),
        )
        win = gui_window.MainWindow(input_data)
        try:
            assert calls == []
            status: str = win._status_label.text()
            assert "parameters.csv (61 values, 4-decimal;" in status
            panel: gui_param_panel.ParamPanel = win._param_panel
            # Without a saved selection the defaults are used, and the
            # initial-state parameters fall back to their midpoints.
            assert panel.selection() == gui_data.default_plot_selection()
            for zone in SPECIATION_ZONES:
                key: str = f"initial_state.{zone}_x"
                lo, hi = input_data.bounds[key]
                assert panel.param_values()[key] == pytest.approx(
                    0.5 * (lo + hi)
                )
            # Geometry and splitter sizes keep their defaults.
            assert win.width() == 1200
            assert win.height() == 900
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            # No saved state carried the toggle: it restores off.
            assert not win._save_history_checkbox.isChecked()
            _assert_restored_dir(win, run_dir)
        finally:
            _close_window(win)

    def test_restores_hydro_only(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hydrology-only saved run restores its mode, no re-run."""
        state = _full_state(input_data, mode=MODE_HYDRO)
        run_dir: Path = _write_saved_run(
            tmp_path, input_data, state, hydro_only=True
        )
        calls: list[np.ndarray] = []
        hydro_calls: list[np.ndarray] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            gui_window,
            "arr_to_results",
            _fake_results_factory(calls, input_data),
        )
        monkeypatch.setattr(
            gui_window,
            "arr_to_hydro_results",
            _fake_results_factory(hydro_calls, input_data, hydro_only=True),
        )
        win = gui_window.MainWindow(input_data)
        try:
            # The restore applies the saved results and mode without
            # running either model.
            assert calls == []
            assert hydro_calls == []
            assert win._mode_checkbox.isChecked()
            panel: gui_param_panel.ParamPanel = win._param_panel
            assert panel.param_values()["proportion.1"] == pytest.approx(0.7)
            win.show()
            _pump()
            # The reactive-transport parts of the panel are hidden.
            assert panel._groups["reactive"].isHidden()
            assert panel._groups["initial"].isHidden()
            assert not panel._groups["hydrology"].isHidden()
            # Streamflow and zone-storage figures, one target column.
            figure: gui_figure_panel.FigurePanel = win._figure_panel
            assert figure.page_count() == 2
            assert figure._titles == ["Streamflow", "Zone storage"]
            assert figure._title_label.text() == "Streamflow"
            assert figure._page_label.text() == "1 / 2"
            assert win._objective_model.columnCount() == 1
            assert win._results is not None
            assert win._results.rt is None
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            assert win._status_label.text() == (
                "Restored the most recent saved run; the figures were "
                "built from its saved output."
            )
            _assert_restored_dir(win, run_dir)
        finally:
            _close_window(win)


# ------------------------------------------------------------------ #
# MainWindow: runs
# ------------------------------------------------------------------ #


class TestMainWindowRuns:
    """Background runs: reruns, failures, and the save button."""

    def test_value_change_starts_second_run(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Editing a value after the first run starts a second one."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump(50)
            win._maybe_start_run()  # nothing to do: the deleted-worker case
            assert len(calls) == 1
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            row._spin.setValue(0.775)
            _wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            assert len(calls) == 2
            assert calls[1][index] == pytest.approx(0.775)
            assert re.fullmatch(
                r"Run 2 finished in \d+\.\d s\.", win._status_label.text()
            )
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
        finally:
            _close_window(win)

    def test_run_history_toggle(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The run-history toggle gates the abbreviated history file."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            history_file: Path = (
                tmp_path / "results" / RUN_HISTORY_FILENAME
            )
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            # The startup run finished with the toggle off: no
            # history file.
            assert not history_file.is_file()

            win._save_history_checkbox.setChecked(True)
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            row._spin.setValue(0.775)
            _wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            _pump(50)
            assert history_file.is_file()
            lines: list[str] = history_file.read_text(
                encoding="utf-8"
            ).splitlines()
            assert len(lines) == 1
            record: dict[str, object] = json.loads(lines[0])
            assert record["mode"] == MODE_FULL
            assert record["param_values"] == win._param_panel.param_values()
            param_values: object = record["param_values"]
            assert isinstance(param_values, dict)
            assert param_values["proportion.1"] == pytest.approx(0.775)
            # Every measured target comes back, with the synthetic
            # objective values.
            expected_objectives: dict[str, dict[str, float]] = {
                "Streamflow": {
                    name: 0.9 * (i + 1)
                    for i, name in enumerate(METRIC_NAMES)
                }
            }
            for species in ("DOC", "DIC", "Ca++"):
                expected_objectives[species] = {
                    name: 0.5 + 0.1 * i
                    for i, name in enumerate(METRIC_NAMES)
                }
            assert record["objectives"] == expected_objectives
            assert datetime.fromisoformat(
                str(record["timestamp"])
            ).tzinfo is not None

            # With the toggle off again the history stays put.
            win._save_history_checkbox.setChecked(False)
            row._spin.setValue(0.825)
            _wait_until(
                lambda: win._run_generation == 3 and win._worker is None
            )
            _pump(50)
            lines = history_file.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 1
        finally:
            _close_window(win)

    def test_failed_run(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A run that raises shows the error and leaves the window empty."""

        def fake(
            x: np.ndarray, _input_data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            raise ValueError("boom")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", fake)
        win = gui_window.MainWindow(input_data)
        try:
            _wait_until(
                lambda: win._worker is None
                and win._status_label.text() == "Run failed: ValueError: boom"
            )
            assert win._results is None
            assert win._last_run_x is None
            assert not win._save_button.isEnabled()
            # No valid results on screen: a manual run can retry.
            assert win._run_button.isEnabled()
            assert win._figure_panel.page_count() == 0
            # No objectives were reported: the table keeps its metric
            # rows but carries no target columns.
            assert win._objective_model.rowCount() == 7
            assert win._objective_model.columnCount() == 0
            win._on_save_run()
            assert win._status_label.text() == (
                "Nothing to save yet: the model has not run."
            )
        finally:
            _close_window(win)

    def test_water_volume_error_modal(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A water-volume failure gets a modal dialog besides the status."""
        message: str = "the zone volume exceeds its maximum"

        def fake(
            x: np.ndarray, _input_data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            raise WaterVolumeError(message)

        monkeypatch.setattr(gui_window, "QMessageBox", _FakeMessageBox)
        _FakeMessageBox.critical_calls.clear()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", fake)
        win = gui_window.MainWindow(input_data)
        try:
            _wait_until(
                lambda: win._worker is None
                and win._status_label.text()
                == f"Run failed: WaterVolumeError: {message}"
            )
            _pump()
            assert _FakeMessageBox.critical_calls == [
                ("Water volume error", message)
            ]
            assert win._results is None
            assert not win._save_button.isEnabled()
            assert win._run_button.isEnabled()
        finally:
            _close_window(win)

    def test_save_run(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save writes the run output plus the session state."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            run_dir: Path = tmp_path / "results" / "run_fake"
            saves: list[dict[str, object]] = []

            def fake_save(
                model: pt.Model,
                hydro: pt.HydroModelResults,
                rt: pt.RtModelResults,
                results_dir: Path | None = None,
            ) -> Path:
                run_dir.mkdir(parents=True, exist_ok=True)
                saves.append(
                    {
                        "model": model,
                        "hydro": hydro,
                        "rt": rt,
                        "results_dir": results_dir,
                    }
                )
                return run_dir

            # Show the window first: the save records the splitter's
            # laid-out sizes, which only exist once Qt has arranged
            # the sections against the window's actual geometry.
            win.show()
            _pump()
            # Showing an offscreen window moves it off the origin;
            # restore the intended client geometry so the saved state
            # is exactly the default the constructor set.
            win.setGeometry(0, 0, *gui_window._INITIAL_SIZE)
            _pump()
            monkeypatch.setattr(gui_window, "save_model_run", fake_save)
            win._on_save_run()
            assert len(saves) == 1
            assert saves[0]["results_dir"] is None
            assert win._status_label.text() == (
                f"Saved run to {run_dir}; 3 figure page(s) saved to "
                f"{run_dir / 'figures'}."
            )
            # The figures are saved as plotted, named after their
            # page titles (non-alphanumerics collapse to a single
            # underscore).
            figure_names: list[str] = sorted(
                path.name for path in (run_dir / "figures").glob("*.png")
            )
            assert figure_names == [
                "Mineral_reaction_rates.png",
                "Streamflow_and_river_concentrations.png",
                "Zone_concentrations_and_storage.png",
            ]

            saved = read_gui_state(run_dir)
            assert saved is not None
            assert saved.csv_path is None
            assert saved.param_values == win._param_panel.param_values()
            assert saved.param_bounds == win._param_panel.current_bounds()
            assert saved.selection == gui_data.default_plot_selection()
            # The window was shown and restored to the default origin
            # and size, so it saves exactly that client geometry; the
            # splitter saves its laid-out sizes, i.e. the initial
            # distribution scaled to the window's height.  Compare
            # proportions rather than raw pixels.
            assert saved.window_geometry == {
                "x": 0,
                "y": 0,
                "width": 1200,
                "height": 900,
            }
            # The saved list holds the main vertical splitter's
            # laid-out sizes: scaled to the splitter's actual size,
            # with each section clamped to its child's minimum size
            # hint, so compare the proportions against the initial
            # split with a loose tolerance.
            assert saved.splitter_sizes is not None
            sizes: list[float] = [float(s) for s in saved.splitter_sizes]
            initial: tuple[int, int] = gui_window._INITIAL_SPLIT_SIZES
            total: float = sum(sizes)
            assert total > 0
            target_total: float = float(sum(initial))
            for size, target in zip(sizes, initial):
                assert size / total == pytest.approx(
                    target / target_total, abs=0.05
                )

            # The run-history toggle is part of the saved state:
            # off here by default, and the second save records it on.
            assert saved.save_history is False
            win._save_history_checkbox.setChecked(True)
            win._on_save_run()
            assert len(saves) == 2
            saved_on = read_gui_state(run_dir)
            assert saved_on is not None
            assert saved_on.save_history is True

            # A parameter change after the run blocks the save (the
            # results on screen would be stale).
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            row._spin.setValue(0.775)
            win._debounce_timer.stop()
            win._on_save_run()
            assert win._status_label.text() == (
                "The model is still catching up; let the current run "
                "finish before saving."
            )
            assert len(saves) == 2
        finally:
            _close_window(win)

    def test_close_with_run_in_flight(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Closing asks to wait; No cancels, Yes pumps until the run ends."""
        releases: list[threading.Event] = []

        def blocking_fake(
            x: np.ndarray, _input_data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            release: threading.Event = threading.Event()
            releases.append(release)
            release.wait()
            return gui_data.GuiRunResults(*_synthetic_results(_input_data))

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", blocking_fake)
        monkeypatch.setattr(gui_window, "QMessageBox", _FakeMessageBox)
        win = gui_window.MainWindow(input_data)
        try:
            # First run in flight: declining cancels the close.
            _wait_until(
                lambda: win._worker is not None and win._worker.isRunning()
            )
            _FakeMessageBox.answer = _FakeMessageBox.StandardButton.No
            event: QCloseEvent = QCloseEvent()
            win.closeEvent(event)
            assert not event.isAccepted()
            assert win._worker is not None and win._worker.isRunning()

            # Let the run finish: the results come up as usual.
            releases[0].set()
            _wait_until(
                lambda: win._worker is None
                and "Run 1 finished" in win._status_label.text()
            )

            # Second run in flight: accepting waits it out.
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            row._spin.setValue(0.775)
            _wait_until(
                lambda: win._run_generation == 2 and len(releases) == 2
            )
            threading.Timer(0.3, releases[1].set).start()
            _FakeMessageBox.answer = _FakeMessageBox.StandardButton.Yes
            event2: QCloseEvent = QCloseEvent()
            win.closeEvent(event2)
            assert event2.isAccepted()
            _wait_until(
                lambda: win._worker is None
                and win._results is not None
            )
            assert win._run_generation == 2
        finally:
            _close_window(win)

    def test_arrow_actions_flip_figure_pages(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The Left / Right actions cycle through the three figures."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            actions: dict[str, QAction] = {
                action.text(): action for action in win.findChildren(QAction)
            }
            assert set(actions) == {"Previous figure", "Next figure"}
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            assert panel.current_page() == 0
            actions["Next figure"].triggered.emit()
            assert panel.current_page() == 1
            assert panel._title_label.text() == "Zone concentrations and storage"
            assert panel._page_label.text() == "2 / 3"
            actions["Next figure"].triggered.emit()
            assert panel.current_page() == 2
            assert panel._title_label.text() == "Mineral reaction rates"
            assert panel._page_label.text() == "3 / 3"
            # Wrap-around in both directions.
            actions["Next figure"].triggered.emit()
            assert panel.current_page() == 0
            assert panel._page_label.text() == "1 / 3"
            actions["Previous figure"].triggered.emit()
            assert panel.current_page() == 2
            assert panel._page_label.text() == "3 / 3"
        finally:
            _close_window(win)

    def test_hydro_only_mode(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Toggling the mode re-runs in the other mode and re-lays out."""
        full_calls: list[np.ndarray] = []
        hydro_calls: list[np.ndarray] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            gui_window,
            "arr_to_results",
            _fake_results_factory(full_calls, input_data),
        )
        monkeypatch.setattr(
            gui_window,
            "arr_to_hydro_results",
            _fake_results_factory(hydro_calls, input_data, hydro_only=True),
        )
        win = gui_window.MainWindow(input_data)
        try:
            # The startup run is in full mode (the checkbox is off).
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert len(full_calls) == 1
            assert hydro_calls == []
            win.show()
            _pump()
            panel: gui_param_panel.ParamPanel = win._param_panel
            assert not panel._groups["reactive"].isHidden()
            assert not panel._groups["initial"].isHidden()
            assert win._figure_panel.page_count() == 3
            assert win._objective_model.columnCount() == 4

            # Switching to hydrology-only re-runs in that mode and
            # hides the reactive-transport parts of the panel.
            win._mode_checkbox.setChecked(True)
            _wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            _pump(50)
            assert len(hydro_calls) == 1
            assert len(full_calls) == 1
            assert np.array_equal(
                hydro_calls[0], _midpoint_vector(input_data)
            )
            assert panel._groups["reactive"].isHidden()
            assert panel._groups["initial"].isHidden()
            # The hidden checkboxes keep their state, so the plot
            # selection is unchanged.
            assert panel.selection() == gui_data.default_plot_selection()
            assert win._figure_panel.page_count() == 2
            assert win._figure_panel._titles == [
                "Streamflow",
                "Zone storage",
            ]
            assert win._objective_model.columnCount() == 1
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()

            # A parameter change in hydro mode records the mode in
            # the run history (Streamflow is the only target).
            win._save_history_checkbox.setChecked(True)
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            row._spin.setValue(0.775)
            _wait_until(
                lambda: win._run_generation == 3 and win._worker is None
            )
            _pump(50)
            assert len(hydro_calls) == 2
            history_file: Path = (
                tmp_path / "results" / RUN_HISTORY_FILENAME
            )
            lines: list[str] = history_file.read_text(
                encoding="utf-8"
            ).splitlines()
            assert len(lines) == 1
            record: dict[str, object] = json.loads(lines[0])
            assert record["mode"] == MODE_HYDRO
            param_values: object = record["param_values"]
            assert isinstance(param_values, dict)
            assert param_values["proportion.1"] == pytest.approx(0.775)
            objectives: object = record["objectives"]
            assert isinstance(objectives, dict)
            assert set(objectives) == {"Streamflow"}

            # Saving a hydrology-only run writes the flat bundle and
            # records the mode in the state file.
            run_dir: Path = tmp_path / "results" / "run_fake"
            saves: list[dict[str, object]] = []

            def fake_hydro_save(
                model: HydrologicalModel,
                hydro: pt.HydroModelResults,
                results_dir: Path | None = None,
            ) -> Path:
                run_dir.mkdir(parents=True, exist_ok=True)
                saves.append(
                    {
                        "model": model,
                        "hydro": hydro,
                        "results_dir": results_dir,
                    }
                )
                return run_dir

            monkeypatch.setattr(
                gui_window, "save_hydro_run", fake_hydro_save
            )
            win._on_save_run()
            assert len(saves) == 1
            assert saves[0]["results_dir"] is None
            assert win._status_label.text() == (
                f"Saved run to {run_dir}; 2 figure page(s) saved to "
                f"{run_dir / 'figures'}."
            )
            saved = read_gui_state(run_dir)
            assert saved is not None
            assert saved.mode == MODE_HYDRO

            # Switching back to full mode re-runs with the edited
            # value and restores the full layout.
            win._mode_checkbox.setChecked(False)
            _wait_until(
                lambda: win._run_generation == 4 and win._worker is None
            )
            _pump(50)
            assert len(full_calls) == 2
            index: int = list(input_data.bounds).index("proportion.1")
            assert full_calls[1][index] == pytest.approx(0.775)
            assert not panel._groups["reactive"].isHidden()
            assert not panel._groups["initial"].isHidden()
            assert win._figure_panel.page_count() == 3
            assert win._objective_model.columnCount() == 4
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
        finally:
            _close_window(win)


# ------------------------------------------------------------------ #
# MainWindow: auto-run toggle
# ------------------------------------------------------------------ #


class TestAutoRunToggle:
    """The auto-run checkbox and the manual run model button."""

    def test_run_button_manual_run_with_auto_run_off(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With auto-run off, edits wait for a click on the run button."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump(50)
            assert win._auto_run_checkbox.isChecked()
            assert not win._run_button.isEnabled()
            win._auto_run_checkbox.setChecked(False)
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            row._spin.setValue(0.775)
            # The results on screen are stale, so the run button is
            # enabled; no auto-run is armed, so nothing else happens.
            assert win._run_button.isEnabled()
            _pump(200)
            assert len(calls) == 1
            assert win._run_button.isEnabled()
            win._run_button.click()
            # The run started synchronously; whether or not it has
            # already finished, the button stays disabled (a run is
            # in flight, and a finished run matches the current
            # values).
            assert win._run_generation == 2
            assert not win._run_button.isEnabled()
            _wait_until(
                lambda: win._worker is None and win._run_generation == 2
            )
            assert len(calls) == 2
            assert calls[1][index] == pytest.approx(0.775)
            assert not win._run_button.isEnabled()
        finally:
            _close_window(win)

    def test_auto_run_off_suppresses_debounced_rerun(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Turning auto-run off before a debounce fires stops the run."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump(50)
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            # Auto-run is on, so the edit arms the debounce; turning
            # the checkbox off before it fires must still stop the
            # run (the gate is checked when the timer fires).
            row._spin.setValue(0.775)
            win._auto_run_checkbox.setChecked(False)
            _wait_until(lambda: not win._debounce_timer.isActive())
            assert len(calls) == 1
            # Re-checking auto-run and editing again starts the run
            # with the latest value as usual.
            win._auto_run_checkbox.setChecked(True)
            row._spin.setValue(0.78)
            _wait_until(
                lambda: win._worker is None and win._run_generation == 2
            )
            assert len(calls) == 2
            assert calls[1][index] == pytest.approx(0.78)
        finally:
            _close_window(win)


# ------------------------------------------------------------------ #
# MainWindow: history
# ------------------------------------------------------------------ #


class TestHistory:
    """The *History* dropdown: restoring and overlaying without a re-run."""

    def test_records_each_run(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each finished run becomes the newest dropdown entry."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            combo = win._param_panel._history_combo
            # No saved run on disk: the dropdown stays empty until
            # the startup run has finished.
            assert combo.count() == 0
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert combo.count() == 1
            assert re.fullmatch(
                r"Run 1 \u2014 \d{2}:\d{2}:\d{2} \(full\)",
                combo.itemText(0),
            )

            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            row._spin.setValue(0.775)
            _wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            assert combo.count() == 2
            assert re.fullmatch(
                r"Run 2 \u2014 \d{2}:\d{2}:\d{2} \(full\)",
                combo.itemText(0),
            )
            assert re.fullmatch(
                r"Run 1 \u2014 \d{2}:\d{2}:\d{2} \(full\)",
                combo.itemText(1),
            )
        finally:
            _close_window(win)

    def test_restore_does_not_rerun(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Restoring a session run replaces values and results, without a re-run."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)

            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            row._spin.setValue(0.775)
            _wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            assert len(calls) == 2

            # The newer run is on screen; restore the first one.
            combo = win._param_panel._history_combo
            assert combo.itemText(0).startswith("Run 2")
            combo.setCurrentIndex(1)
            win._on_history_restore()

            # The restore rebuilds the panel's cells; the fresh one
            # for this parameter carries the restored value.
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            assert row.value() == pytest.approx(0.6)
            assert len(calls) == 2
            assert win._run_generation == 2
            last_x: np.ndarray | None = win._last_run_x
            assert last_x is not None
            assert np.array_equal(last_x, _midpoint_vector(input_data))
            assert last_x[index] == pytest.approx(0.6)
            assert win._last_run_mode == MODE_FULL
            results: gui_data.GuiRunResults | None = win._results
            assert results is not None
            assert results.rt is not None
            assert not win._run_button.isEnabled()
            assert win._status_label.text() == (
                f"Restored the previous run {combo.itemText(1)}."
            )
        finally:
            _close_window(win)

    def test_restore_switches_mode(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Restoring a run of the other mode re-lays-out the window, without a re-run."""
        full_calls: list[np.ndarray] = []
        hydro_calls: list[np.ndarray] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            gui_window,
            "arr_to_results",
            _fake_results_factory(full_calls, input_data),
        )
        monkeypatch.setattr(
            gui_window,
            "arr_to_hydro_results",
            _fake_results_factory(hydro_calls, input_data, hydro_only=True),
        )
        win = gui_window.MainWindow(input_data)
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert len(full_calls) == 1

            win._mode_checkbox.setChecked(True)
            _wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            _pump(50)
            assert len(hydro_calls) == 1

            # Restore the first (full) run.
            combo = win._param_panel._history_combo
            combo.setCurrentIndex(1)
            win._on_history_restore()
            win.show()
            _pump()
            assert not win._mode_checkbox.isChecked()
            assert win._last_run_mode == MODE_FULL
            results: gui_data.GuiRunResults | None = win._results
            assert results is not None
            assert results.rt is not None
            assert not win._param_panel._groups["reactive"].isHidden()
            assert not win._param_panel._groups["initial"].isHidden()
            assert win._figure_panel.page_count() == 3
            assert win._objective_model.columnCount() == 4
            assert len(full_calls) == 1
            assert len(hydro_calls) == 1

            # Restore the second (hydrology-only) run.
            combo.setCurrentIndex(0)
            win._on_history_restore()
            _pump()
            assert win._mode_checkbox.isChecked()
            assert win._last_run_mode == MODE_HYDRO
            results = win._results
            assert results is not None
            assert results.rt is None
            assert win._param_panel._groups["reactive"].isHidden()
            assert win._param_panel._groups["initial"].isHidden()
            assert win._figure_panel.page_count() == 2
            assert win._objective_model.columnCount() == 1
            assert len(full_calls) == 1
            assert len(hydro_calls) == 1
        finally:
            _close_window(win)

    def test_overlay_no_rerun(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The overlay checkbox redraws the figures from the selected run, without a re-run."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert len(calls) == 1

            panel = win._param_panel
            combo = panel._history_combo
            assert combo.count() == 1
            assert combo.currentIndex() == 0

            assert not panel._history_overlay_checkbox.isChecked()
            panel._history_overlay_checkbox.setChecked(True)
            _pump()

            # Page 0 is the streamflow figure: the previous run's
            # line sits behind the current one, half transparent.
            figure: Figure = win._figure_panel._canvases[0].figure
            hydro_lines = _lines(figure.axes[0])
            assert len(hydro_lines) == 2
            assert hydro_lines[0].get_alpha() == 0.5
            assert _alpha(hydro_lines[1]) == 1.0
            assert hydro_lines[1].get_label() == "Streamflow (sim)"
            # The legend still lists only the current run's series.
            assert len(figure.legends[0].get_texts()) == 8

            # Unchecking the overlay removes the previous run's lines.
            panel._history_overlay_checkbox.setChecked(False)
            _pump()
            figure = win._figure_panel._canvases[0].figure
            assert len(_lines(figure.axes[0])) == 1
            assert len(calls) == 1
        finally:
            _close_window(win)

    def test_restore_empty_history(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A restore with an empty dropdown reports that there is nothing to restore."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            # The startup run is still in flight: no history yet, and
            # a restore says exactly that.
            assert win._param_panel._history_combo.count() == 0
            win._on_history_restore()
            assert win._status_label.text() == (
                "No previous run selected: the history is empty."
            )
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert win._param_panel._history_combo.count() == 1
        finally:
            _close_window(win)

    def test_saved_dir_restore(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A saved run directory joins the dropdown and restores from disk, without a re-run."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)

            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            row._spin.setValue(0.775)
            _wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            assert len(calls) == 2

            # Save the current run for real: the new directory joins
            # the dropdown below the session's runs.
            win._on_save_run()
            status: str = win._status_label.text()
            match: re.Match[str] | None = re.fullmatch(
                r"Saved run to (?P<dir>[^;]+); \d+ figure page\(s\) "
                r"saved to .*",
                status,
            )
            assert match is not None
            run_dir: Path = Path(match.group("dir"))
            assert run_dir.is_dir()
            assert read_gui_state(run_dir) is not None

            combo = win._param_panel._history_combo
            assert combo.count() == 3
            saved_index: int = combo.findText(run_dir.name)
            assert saved_index == 2

            # A third run, then restore the saved directory from disk.
            row._spin.setValue(0.825)
            _wait_until(
                lambda: win._run_generation == 3 and win._worker is None
            )
            saved_index = combo.findText(run_dir.name)
            assert saved_index == 3
            combo.setCurrentIndex(saved_index)
            win._on_history_restore()

            # The restore rebuilds the panel's cells; the fresh one
            # for this parameter carries the restored value.
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            assert row.value() == pytest.approx(0.775)
            last_x: np.ndarray | None = win._last_run_x
            assert last_x is not None
            assert last_x[index] == pytest.approx(0.775)
            assert win._last_run_mode == MODE_FULL
            results: gui_data.GuiRunResults | None = win._results
            assert results is not None
            assert results.rt is not None
            assert results.hydro.saved_dir == str(run_dir)
            assert len(calls) == 3
            assert not win._run_button.isEnabled()
            assert win._status_label.text() == (
                f"Restored the previous run {run_dir.name}."
            )
        finally:
            _close_window(win)


# ------------------------------------------------------------------ #
# FigurePanel (standalone)
# ------------------------------------------------------------------ #


def _figure(title: str) -> tuple[str, Figure]:
    """A trivial single-axis figure under the given title."""
    fig: Figure = Figure()
    fig.suptitle(title)
    fig.add_subplot()
    return title, fig


class TestFigurePanel:
    """The figure panel's paging and header behavior."""

    def test_empty_panel(self, qapp: QApplication) -> None:
        panel = gui_figure_panel.FigurePanel()
        try:
            assert panel.page_count() == 0
            assert panel.current_page() == -1
            assert panel._title_label.text() == ""
            assert panel._page_label.text() == ""
            assert not panel._prev_button.isEnabled()
            assert not panel._next_button.isEnabled()
        finally:
            panel.deleteLater()
            _pump()

    def test_paging_and_wrap_around(self, qapp: QApplication) -> None:
        panel = gui_figure_panel.FigurePanel()
        try:
            panel.set_figures(
                [
                    _figure("A"),
                    _figure("B"),
                    _figure("C"),
                ]
            )
            assert panel.page_count() == 3
            assert panel.current_page() == 0
            assert panel._title_label.text() == "A"
            assert panel._page_label.text() == "1 / 3"
            assert panel._prev_button.isEnabled()
            assert panel._next_button.isEnabled()
            panel.next_page()
            assert panel.current_page() == 1
            assert panel._title_label.text() == "B"
            assert panel._page_label.text() == "2 / 3"
            panel.next_page()
            assert panel.current_page() == 2
            assert panel._page_label.text() == "3 / 3"
            panel.next_page()  # wraps to the first page
            assert panel.current_page() == 0
            panel.previous_page()  # wraps to the last page
            assert panel.current_page() == 2
            panel.show_page(99)  # clamped to the last page
            assert panel.current_page() == 2
            panel.show_page(-3)  # clamped to the first page
            assert panel.current_page() == 0
        finally:
            panel.deleteLater()
            _pump()

    def test_set_figures_clamps_the_current_page(
        self, qapp: QApplication
    ) -> None:
        """Shrinking the figure list keeps the index valid."""
        panel = gui_figure_panel.FigurePanel()
        try:
            panel.set_figures([_figure("A"), _figure("B"), _figure("C")])
            panel.show_page(1)
            panel.set_figures([_figure("A"), _figure("B")])
            assert panel.current_page() == 1
            assert panel._page_label.text() == "2 / 2"
            panel.set_figures([_figure("A")])
            assert panel.current_page() == 0
            assert panel._page_label.text() == "1 / 1"
            assert not panel._prev_button.isEnabled()
            assert not panel._next_button.isEnabled()
            panel.set_figures(())
            assert panel.page_count() == 0
            assert panel.current_page() == -1
            assert panel._title_label.text() == ""
        finally:
            panel.deleteLater()
            _pump()

    def test_save_pages(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        """Every page is saved as a PNG named after its title."""
        panel = gui_figure_panel.FigurePanel()
        try:
            panel.set_figures(
                [
                    _figure("Streamflow and river concentrations"),
                    _figure("Mineral reaction rates"),
                    _figure("!!!"),
                ]
            )
            out_dir: Path = tmp_path / "nested" / "figures"
            paths: list[Path] = panel.save_pages(out_dir)
            # Each title's non-alphanumerics collapse to a single
            # underscore; a title that sanitizes to nothing falls
            # back to the page's position.
            assert [path.name for path in paths] == [
                "Streamflow_and_river_concentrations.png",
                "Mineral_reaction_rates.png",
                "figure_3.png",
            ]
            # The directory is created with its parents.
            for path in paths:
                assert path.parent == out_dir
                assert path.is_file()
        finally:
            panel.deleteLater()
            _pump()


# ------------------------------------------------------------------ #
# _ObjectiveTableModel (standalone)
# ------------------------------------------------------------------ #


def _objective_table() -> pd.DataFrame:
    """A two-column objective table with NaNs in the second column."""
    values: dict[str, list[float]] = {
        "Streamflow": [0.9, 0.5, 0.8, 0.4, 0.01, 0.7, 0.6],
        "DOC": [float("nan")] * 7,
    }
    return pd.DataFrame(values, index=list(METRIC_LABELS.values()))


class TestObjectiveTableModel:
    """The model behind the table panel left of the parameter panel."""

    def test_empty_model(self, qapp: QApplication) -> None:
        model = gui_window._ObjectiveTableModel()
        try:
            # With no objectives yet the table keeps its metric rows
            # but carries no target columns.
            assert model.rowCount() == 7
            assert model.columnCount() == 0
            headers = [
                model.headerData(i, Qt.Orientation.Vertical)
                for i in range(7)
            ]
            assert headers == list(METRIC_LABELS.values())
        finally:
            model.deleteLater()
            _pump()

    def test_set_table(self, qapp: QApplication) -> None:
        model = gui_window._ObjectiveTableModel()
        try:
            model.set_table(_objective_table())
            assert model.rowCount() == 7
            assert model.columnCount() == 2
            column_headers = [
                model.headerData(i, Qt.Orientation.Horizontal)
                for i in range(2)
            ]
            assert column_headers == ["Streamflow", "DOC"]
            row_headers = [
                model.headerData(i, Qt.Orientation.Vertical)
                for i in range(7)
            ]
            assert row_headers == list(METRIC_LABELS.values())
            assert model.data(model.index(0, 0)) == "0.900"
            assert model.data(model.index(0, 1)) == "–"
            # Invalid indices and foreign roles yield None.
            assert model.data(model.index(9, 0)) is None
            assert (
                model.data(
                    model.index(0, 0), Qt.ItemDataRole.ToolTipRole
                )
                is None
            )
        finally:
            model.deleteLater()
            _pump()


# ------------------------------------------------------------------ #
# ParamPanel (standalone)
# ------------------------------------------------------------------ #


class TestParamPanel:
    """Free cells, pinned cells, their dialogs, and the plot selection."""

    @staticmethod
    def _make_panel(
        input_data: gui_data.GuiInputData,
    ) -> gui_param_panel.ParamPanel:
        return gui_param_panel.ParamPanel(
            input_data, gui_data.default_plot_selection()
        )

    def test_construction_is_silent(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        panel = self._make_panel(input_data)
        try:
            values: list[str] = []
            bounds: list[tuple[str, float, float]] = []
            selections: list[PlotSelection] = []
            panel.value_changed.connect(values.append)
            panel.bounds_changed.connect(
                lambda k, lo, hi: bounds.append((k, lo, hi))
            )
            panel.selection_changed.connect(selections.append)
            assert values == []
            assert bounds == []
            assert selections == []
            assert panel.selection() == gui_data.default_plot_selection()
        finally:
            panel.deleteLater()
            _pump()

    def test_free_cell_value(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        panel = self._make_panel(input_data)
        try:
            row = panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel._FreeCell)
            values: list[str] = []
            panel.value_changed.connect(values.append)
            row._spin.setValue(0.775)
            assert values == ["proportion.1"]
            assert row.value() == pytest.approx(0.775)
            assert panel.param_values()["proportion.1"] == pytest.approx(0.775)
            # set_value is for programmatic updates: no signal, and
            # the panel's copy of the value is left for the caller.
            row.set_value(0.3)
            assert values == ["proportion.1"]
            assert row.value() == pytest.approx(0.3)
            assert panel.param_values()["proportion.1"] == pytest.approx(0.775)
        finally:
            panel.deleteLater()
            _pump()

    def test_cell_layout(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        """Each cell places the symbol left of its value, a stretch between."""
        panel = self._make_panel(input_data)
        try:
            free = panel._rows["proportion.1"]
            assert isinstance(free, gui_param_panel._FreeCell)
            layout = free.layout()
            assert isinstance(layout, QHBoxLayout)
            assert layout.count() == 3
            symbol_item = layout.itemAt(0)
            assert symbol_item is not None
            assert symbol_item.widget() is free._symbol
            stretch_item = layout.itemAt(1)
            assert stretch_item is not None
            assert stretch_item.spacerItem() is not None
            spin_item = layout.itemAt(2)
            assert spin_item is not None
            assert spin_item.widget() is free._spin
            # The spin box is frameless: the cell's border, drawn by
            # the group, is the cell's only frame, so the grid lines
            # stay continuous.
            assert "border: none" in free._spin.styleSheet()

            pinned = panel._rows["ground_hs.alpha"]
            assert isinstance(pinned, gui_param_panel._PinnedCell)
            pinned_layout = pinned.layout()
            assert isinstance(pinned_layout, QHBoxLayout)
            assert pinned_layout.count() == 3
            symbol_item = pinned_layout.itemAt(0)
            assert symbol_item is not None
            assert symbol_item.widget() is pinned._symbol
            stretch_item = pinned_layout.itemAt(1)
            assert stretch_item is not None
            assert stretch_item.spacerItem() is not None
            value_item = pinned_layout.itemAt(2)
            assert value_item is not None
            assert value_item.widget() is pinned._value
        finally:
            panel.deleteLater()
            _pump()

    def test_grid_cell_banding(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        """Themed cells carry the grid's border and alternating row bands."""
        panel = self._make_panel(input_data)
        try:
            for name, theme in (
                ("hydrology", HYDRO_THEME),
                ("reactive", RT_THEME),
            ):
                group: gui_param_panel.ThemedGroup = panel._groups[name]
                assert group.theme == theme
                band_a, band_b, line = gui_param_panel._GRID_STYLES[
                    theme.name
                ]

                cells: list[QWidget] = group.content().findChildren(
                    gui_param_panel._FreeCell
                ) + group.content().findChildren(
                    gui_param_panel._PinnedCell
                )
                cells_by_row: dict[int, list[QWidget]] = {}
                for cell in cells:
                    row, _col, _span = group._grid_positions[id(cell)]
                    cells_by_row.setdefault(row, []).append(cell)

                row_bands: list[str] = []
                for row in sorted(cells_by_row):
                    row_cells: list[QWidget] = cells_by_row[row]
                    for cell in row_cells:
                        assert cell.objectName().startswith("grid_cell_")
                        assert cell.testAttribute(
                            Qt.WidgetAttribute.WA_StyledBackground
                        )
                        assert f"border: 1px solid {line}" in cell.styleSheet()
                        assert _band_of(cell) == _band_of(row_cells[0])
                    row_bands.append(_band_of(row_cells[0]))

                # The cell rows alternate between the theme's two
                # bands and repeat every two rows; the zone and
                # mineral headers between the rows do not disturb
                # the alternation.
                assert row_bands[0] == band_a
                for first, second in zip(row_bands, row_bands[1:]):
                    assert first != second
                for first, third in zip(row_bands, row_bands[2:]):
                    assert first == third
                assert set(row_bands) == {band_a, band_b}

                # The full-width zone headers stay undecorated.
                headers: list[QLabel] = group.content().findChildren(
                    gui_param_panel._ZoneHeader
                )
                assert len(headers) >= 2
                for header in headers:
                    assert header.styleSheet() == ""
                    assert header.objectName() == ""

            # The single-row groups take the first band.
            for name in ("proportion", "initial"):
                single_group: gui_param_panel.ThemedGroup = panel._groups[
                    name
                ]
                assert single_group.theme is not None
                expected_band: str = gui_param_panel._GRID_STYLES[
                    single_group.theme.name
                ][0]
                single_cells: list[QWidget] = (
                    single_group.content().findChildren(
                        gui_param_panel._FreeCell
                    )
                    + single_group.content().findChildren(
                        gui_param_panel._PinnedCell
                    )
                )
                assert single_cells
                for single_cell in single_cells:
                    assert _band_of(single_cell) == expected_band

            # The neutral group's widgets keep the default look.
            box: QCheckBox = panel._time_series_boxes["Streamflow"]
            assert box.styleSheet() == ""
            assert box.objectName() == ""
        finally:
            panel.deleteLater()
            _pump()

    def test_unlock_restyles_cell(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unlocked parameter takes over the row's band and border."""
        panel = self._make_panel(input_data)
        try:
            key: str = "ground_hs.alpha"
            old = panel._rows[key]
            assert isinstance(old, gui_param_panel._PinnedCell)
            old_band: str = _band_of(old)
            _install_bounds_stub(monkeypatch, (0.25, 1.75))
            qapp.sendEvent(old._value, _double_click_event())
            new_cell = panel._rows[key]
            assert isinstance(new_cell, gui_param_panel._FreeCell)
            assert new_cell.objectName().startswith("grid_cell_")
            assert new_cell.testAttribute(
                Qt.WidgetAttribute.WA_StyledBackground
            )
            assert _band_of(new_cell) == old_band
        finally:
            panel.deleteLater()
            _pump()

    def test_unlock_pinned_parameter(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Double-clicking a pinned cell gives it a range and a spin box."""
        panel = self._make_panel(input_data)
        try:
            key: str = "ground_hs.alpha"
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel._PinnedCell)
            records = _install_bounds_stub(monkeypatch, (0.25, 1.75))
            bounds: list[tuple[str, float, float]] = []
            values: list[str] = []
            panel.bounds_changed.connect(
                lambda k, lo, hi: bounds.append((k, lo, hi))
            )
            panel.value_changed.connect(values.append)
            qapp.sendEvent(row._value, _double_click_event())
            assert len(records) == 1
            label: str = param_label(key, input_data.network.mineral_names)
            assert records[0]["title"] == f"Edit range — {label}"
            assert records[0]["hint"] == (
                f"{label} is pinned to 1.0000.  Choose a range to make it "
                "adjustable; the current value is kept."
            )
            # The dialog is prefilled with the value's ± half-span.
            assert records[0]["lo"] == pytest.approx(0.5)
            assert records[0]["hi"] == pytest.approx(1.5)
            new_cell = panel._rows[key]
            assert isinstance(new_cell, gui_param_panel._FreeCell)
            assert new_cell._symbol.text() == "alpha"
            assert new_cell._symbol.toolTip() == label
            assert new_cell.value() == pytest.approx(1.0)
            assert new_cell._spin.value() == pytest.approx(1.0)
            assert new_cell._bounds == (0.25, 1.75)
            assert bounds == [(key, 0.25, 1.75)]
            assert values == [key]
        finally:
            panel.deleteLater()
            _pump()

    def test_unlock_pinned_parameter_cancelled(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancelling the dialog leaves the row pinned."""
        panel = self._make_panel(input_data)
        try:
            key: str = "ground_rp.alpha"
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel._PinnedCell)
            records = _install_bounds_stub(monkeypatch, None)
            bounds: list[tuple[str, float, float]] = []
            values: list[str] = []
            panel.bounds_changed.connect(
                lambda k, lo, hi: bounds.append((k, lo, hi))
            )
            panel.value_changed.connect(values.append)
            row.mouseDoubleClickEvent(_double_click_event())
            assert len(records) == 1
            assert isinstance(panel._rows[key], gui_param_panel._PinnedCell)
            assert bounds == []
            assert values == []
        finally:
            panel.deleteLater()
            _pump()

    def test_edit_bounds(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Double-clicking a free cell edits both ends of its range."""
        panel = self._make_panel(input_data)
        try:
            key: str = "proportion.1"
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel._FreeCell)
            bounds: list[tuple[str, float, float]] = []
            values: list[str] = []
            panel.bounds_changed.connect(
                lambda k, lo, hi: bounds.append((k, lo, hi))
            )
            panel.value_changed.connect(values.append)

            records = _install_bounds_stub(monkeypatch, (0.5, 0.95))
            line_edit = row._spin.lineEdit()
            assert line_edit is not None
            qapp.sendEvent(line_edit, _double_click_event())
            assert len(records) == 1
            assert records[0]["title"] == (
                "Edit range — Distal proportion of catchment"
            )
            assert records[0]["hint"] == "Current range: 0.25000 to 0.95000."
            assert records[0]["lo"] == pytest.approx(0.25)
            assert records[0]["hi"] == pytest.approx(0.95)
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel._FreeCell)
            assert row._bounds == (0.5, 0.95)
            assert row.value() == pytest.approx(0.6)  # the value is kept
            assert bounds == [(key, 0.5, 0.95)]
            assert values == [key]

            # The same for the edited range.
            records = _install_bounds_stub(monkeypatch, (0.5, 0.8))
            line_edit = row._spin.lineEdit()
            assert line_edit is not None
            qapp.sendEvent(line_edit, _double_click_event())
            assert records[0]["hint"] == "Current range: 0.50000 to 0.95000."
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel._FreeCell)
            assert row._bounds == (0.5, 0.8)
            assert bounds == [(key, 0.5, 0.95), (key, 0.5, 0.8)]
            assert values == [key, key]
        finally:
            panel.deleteLater()
            _pump()

    def test_bounds_dialog_static_option_pins_parameter(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A degenerate dialog result pins a free parameter at its
        displayed value; a degenerate result from the unlock dialog
        is a no-op; a real range frees the parameter again.
        """
        panel = self._make_panel(input_data)
        try:
            key: str = "proportion.1"
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel._FreeCell)
            v: float = panel._values[key]
            lo, hi = panel._bounds[key]
            bounds: list[tuple[str, float, float]] = []
            values: list[str] = []
            panel.bounds_changed.connect(
                lambda k, lo2, hi2: bounds.append((k, lo2, hi2))
            )
            panel.value_changed.connect(values.append)

            # 1. A degenerate (equal) result from the range dialog
            #    pins the parameter at its displayed value.
            records = _install_bounds_stub(monkeypatch, (v, v))
            line_edit = row._spin.lineEdit()
            assert line_edit is not None
            qapp.sendEvent(line_edit, _double_click_event())
            assert len(records) == 1
            assert records[0]["value"] == pytest.approx(v)
            assert records[0]["lo"] == pytest.approx(lo)
            assert records[0]["hi"] == pytest.approx(hi)
            pinned = panel._rows[key]
            assert isinstance(pinned, gui_param_panel._PinnedCell)
            assert panel._bounds[key] == (v, v)
            assert panel._values[key] == pytest.approx(v)
            assert bounds == [(key, v, v)]
            assert values == [key]

            # 2. A degenerate result from the pinned cell's unlock
            #    dialog is a no-op: the row stays pinned at the
            #    same value.
            records = _install_bounds_stub(monkeypatch, (v, v))
            qapp.sendEvent(pinned._value, _double_click_event())
            assert len(records) == 1
            assert isinstance(panel._rows[key], gui_param_panel._PinnedCell)
            assert panel._bounds[key] == (v, v)
            assert panel._values[key] == pytest.approx(v)
            assert bounds == [(key, v, v)]
            assert values == [key]

            # 3. A real range from the unlock dialog converts the
            #    cell back into a free cell, keeping the value.
            lo2: float = v - 0.2
            hi2: float = v + 0.2
            records = _install_bounds_stub(monkeypatch, (lo2, hi2))
            pinned = panel._rows[key]
            qapp.sendEvent(pinned._value, _double_click_event())
            assert len(records) == 1
            free_cell = panel._rows[key]
            assert isinstance(free_cell, gui_param_panel._FreeCell)
            assert panel._bounds[key] == (lo2, hi2)
            assert panel._values[key] == pytest.approx(v)
            assert bounds == [(key, v, v), (key, lo2, hi2)]
            assert values == [key, key]
        finally:
            panel.deleteLater()
            _pump()

    def test_plot_selection(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        """Toggling a box emits the recomputed selection; set_selection is silent."""
        panel = self._make_panel(input_data)
        try:
            selections: list[PlotSelection] = []
            panel.selection_changed.connect(selections.append)
            panel._species_boxes["HCO3-"].setChecked(True)
            assert len(selections) == 1
            selection = selections[0]
            assert selection.species == ["DOC", "HCO3-"]
            assert selection.zones == SPECIATION_ZONES
            assert selection.minerals == ["SOC(s)"]
            assert selection.time_series == list(TIME_SERIES_OPTIONS)

            # set_selection updates the boxes without emitting and
            # ignores names the network does not know.
            panel.set_selection(
                PlotSelection(
                    species=["DOC", "Bogus"],
                    zones=SPECIATION_ZONES,
                    minerals=["SOC(s)"],
                    time_series=["Streamflow", "DIC"],
                )
            )
            assert len(selections) == 1
            assert panel.selection().species == ["DOC"]
            assert panel.selection().time_series == ["Streamflow", "DIC"]
        finally:
            panel.deleteLater()
            _pump()

    def test_time_series_boxes(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        """Toggling the time-series boxes re-emits in option order."""
        panel = self._make_panel(input_data)
        try:
            selections: list[PlotSelection] = []
            panel.selection_changed.connect(selections.append)
            assert [
                box.isChecked() for box in panel._time_series_boxes.values()
            ] == [True, True, True, True]

            panel._time_series_boxes["DIC"].setChecked(False)
            assert len(selections) == 1
            assert selections[0].time_series == ["Streamflow", "DOC", "Ca++"]

            panel._time_series_boxes["Streamflow"].setChecked(False)
            assert len(selections) == 2
            assert selections[1].time_series == ["DOC", "Ca++"]
        finally:
            panel.deleteLater()
            _pump()

    def test_reset_bounds(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resetting restores the loaded bounds and re-pins the pinned."""
        panel = self._make_panel(input_data)
        try:
            key: str = "ground_hs.alpha"
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel._PinnedCell)
            _install_bounds_stub(monkeypatch, (0.25, 1.75))
            row.mouseDoubleClickEvent(_double_click_event())
            assert isinstance(
                panel._rows[key], gui_param_panel._FreeCell
            )
            panel._values["proportion.1"] = 1.5  # outside the range
            restored: dict[str, float] = panel.reset_bounds()
            assert panel.current_bounds() == dict(input_data.bounds)
            assert restored["proportion.1"] == pytest.approx(0.95)
            assert isinstance(panel._rows[key], gui_param_panel._PinnedCell)
        finally:
            panel.deleteLater()
            _pump()

    def test_group_lookup_and_themes(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        panel = self._make_panel(input_data)
        try:
            with pytest.raises(KeyError):
                panel._group_of("bogus.key")
            assert panel._groups["hydrology"].theme == HYDRO_THEME
            assert panel._plots_group.theme is None

            for name in ("hydrology", "proportion", "reactive", "initial"):
                assert isinstance(
                    panel._groups[name]._content_layout, QGridLayout
                )
                assert (
                    panel._groups[name]._grid_columns
                    == gui_param_panel._GRID_COLUMNS
                )
            assert isinstance(
                panel._plots_group._content_layout, QGridLayout
            )
            assert panel._plots_group._grid_columns == 2

            # Every group's title is left-aligned: the toggle is
            # sized to fit its text (`QToolButton` ignores the
            # `text-align` style property, so a stretched button
            # would center the title) and sits at the left end of a
            # header row, with a 4px left padding matching the
            # cells' left margin.
            def _assert_fitted_toggle(
                group: gui_param_panel.ThemedGroup,
            ) -> None:
                assert (
                    group._toggle.sizePolicy().horizontalPolicy()
                    == QSizePolicy.Policy.Preferred
                )
                assert (
                    group._toggle.sizePolicy().verticalPolicy()
                    == QSizePolicy.Policy.Fixed
                )
                group_layout = group.layout()
                assert group_layout is not None
                header_item = group_layout.itemAt(0)
                assert isinstance(header_item, QLayoutItem)
                header_row = header_item.layout()
                assert isinstance(header_row, QHBoxLayout)
                assert header_row.itemAt(0).widget() is group._toggle
                assert header_row.itemAt(1).spacerItem() is not None

            for name in ("hydrology", "proportion", "reactive", "initial"):
                _assert_fitted_toggle(panel._groups[name])
                assert (
                    "padding: 4px 0 4px 4px;" in panel._groups[name].styleSheet()
                )
            _assert_fitted_toggle(panel._plots_group)
            assert (
                panel._plots_group._toggle.styleSheet()
                == "QToolButton { padding: 4px 0 4px 4px; }"
            )

            group: gui_param_panel.ThemedGroup = panel._groups["hydrology"]
            assert group._toggle.text().startswith("▾")
            assert not group.content().isHidden()
            group._toggle.click()  # collapse
            assert not group._toggle.isChecked()
            assert group.content().isHidden()
            assert group._toggle.text().startswith("▸")
            group._toggle.click()  # expand
            assert group._toggle.isChecked()
            assert not group.content().isHidden()

            with pytest.raises(ValueError, match="not in this group"):
                panel._plots_group.replace_widget(QWidget(), QWidget())
        finally:
            panel.deleteLater()
            _pump()


# ------------------------------------------------------------------ #
# The header's *Gradients* button: the finite-difference jobs
# ------------------------------------------------------------------ #


def _free_keys(input_data: gui_data.GuiInputData) -> list[str]:
    """The keys of the parameters whose loaded range has a width."""
    return [
        key for key, (low, high) in input_data.bounds.items() if high > low
    ]


def _gradient_fake_factory(
    calls: list[np.ndarray],
    input_data: gui_data.GuiInputData,
    hydro_term_key: str,
    rt_term_key: str,
) -> tuple[
    Callable[[np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults],
    Callable[[np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults],
]:
    """
    Stand-ins for `arr_to_results` and `arr_to_hydro_results` that
    record every parameter vector and shift the synthetic
    objectives by ``2 * x[hydro_term_key]`` (the streamflow
    objectives, in both modes) and, in full mode, by ``2 *
    x[rt_term_key]`` (the river species' objectives).  The
    gradient job's central differences then come out exactly: the
    derivative with respect to the two chosen parameters is 2.0
    (in the blocks each shift touches) and 0.0 for every other
    free parameter.
    """
    keys: list[str] = list(input_data.bounds)
    hydro_index: int = keys.index(hydro_term_key)
    rt_index: int = keys.index(rt_term_key)

    def make(
        x: np.ndarray,
        _input_data: gui_data.GuiInputData,
        hydro_only: bool,
    ) -> gui_data.GuiRunResults:
        calls.append(np.array(x))
        hydro, rt = _synthetic_results(input_data)
        hydro.objective_functions = (
            hydro.objective_functions
            + 2.0 * float(x[hydro_index])
        )
        if hydro_only:
            return gui_data.GuiRunResults(hydro, None)
        rt_objectives: pd.DataFrame | None = rt.objective_functions
        assert rt_objectives is not None
        rt.objective_functions = (
            rt_objectives + 2.0 * float(x[rt_index])
        )
        return gui_data.GuiRunResults(hydro, rt)

    def full_fake(
        x: np.ndarray, _input_data: gui_data.GuiInputData
    ) -> gui_data.GuiRunResults:
        return make(x, _input_data, False)

    def hydro_fake(
        x: np.ndarray, _input_data: gui_data.GuiInputData
    ) -> gui_data.GuiRunResults:
        return make(x, _input_data, True)

    return full_fake, hydro_fake


def _gradient_window(
    input_data: gui_data.GuiInputData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    full_fake: Callable[
        [np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults
    ],
    hydro_fake: Callable[
        [np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults
    ],
) -> gui_window.MainWindow:
    """
    Build a main window in an empty `tmp_path` with `full_fake` and
    `hydro_fake` patched in for `arr_to_results` and
    `arr_to_hydro_results` (the gradient job's worker calls both).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gui_window, "arr_to_results", full_fake)
    monkeypatch.setattr(gui_window, "arr_to_hydro_results", hydro_fake)
    return gui_window.MainWindow(input_data)


def _only_csv(tmp_path: Path) -> Path:
    """The single ``gradients_*.csv`` in `tmp_path`'s results directory."""
    csvs: list[Path] = list((tmp_path / "results").glob("gradients_*.csv"))
    assert len(csvs) == 1
    return csvs[0]


class TestGradients:
    """The header's *Gradients* button: the finite-difference jobs."""

    def test_full_mode_job(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        win = _gradient_window(
            input_data, tmp_path, monkeypatch, full_fake, hydro_fake
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            win._on_gradients_clicked()
            _wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            _pump()
            assert len(calls) == 1 + 2 * len(free)

            loaded: pd.DataFrame = pd.read_csv(
                _only_csv(tmp_path), index_col="objective"
            )
            assert loaded.shape == (4 * len(METRIC_NAMES), len(free))
            assert list(loaded.columns) == free
            for i in range(len(METRIC_NAMES)):
                # The streamflow block only feels the hydro shift.
                assert loaded.iloc[i, 0] == pytest.approx(2.0)
                assert loaded.iloc[i, 1] == pytest.approx(0.0)
            for i in range(
                len(METRIC_NAMES), 4 * len(METRIC_NAMES)
            ):
                # The species blocks only feel the rt shift.
                assert loaded.iloc[i, 0] == pytest.approx(0.0)
                assert loaded.iloc[i, 1] == pytest.approx(2.0)
            for i in range(4 * len(METRIC_NAMES)):
                for j in range(2, len(free)):
                    assert loaded.iloc[i, j] == pytest.approx(0.0)

            assert win._figure_panel.page_count() == 4
            assert win._figure_panel._titles[-1] == "Gradients (full)"
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            assert win._gradients_button.isEnabled()
        finally:
            _close_window(win)

    def test_hydro_mode_job(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        win = _gradient_window(
            input_data, tmp_path, monkeypatch, full_fake, hydro_fake
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._mode_checkbox.setChecked(True)
            _wait_until(
                lambda: win._worker is None
                and win._last_run_mode == MODE_HYDRO
                and win._results_are_up_to_date()
            )
            _pump()
            assert len(calls) == 2

            win._on_gradients_clicked()
            _wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            _pump()
            assert len(calls) == 2 + 2 * len(free)

            loaded: pd.DataFrame = pd.read_csv(
                _only_csv(tmp_path), index_col="objective"
            )
            assert loaded.shape == (len(METRIC_NAMES), len(free))
            assert list(loaded.columns) == free
            for i in range(len(METRIC_NAMES)):
                assert loaded.iloc[i, 0] == pytest.approx(2.0)
                assert loaded.iloc[i, 1] == pytest.approx(0.0)
                for j in range(2, len(free)):
                    assert loaded.iloc[i, j] == pytest.approx(0.0)

            assert win._figure_panel.page_count() == 3
            assert win._figure_panel._titles[-1] == "Gradients (hydro)"
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            assert win._gradients_button.isEnabled()
        finally:
            _close_window(win)

    def test_failed_job(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[np.ndarray] = []

        def flaky(
            x: np.ndarray, _input_data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            calls.append(np.array(x))
            if len(calls) > 1:
                raise RuntimeError("boom")
            hydro, rt = _synthetic_results(input_data)
            return gui_data.GuiRunResults(hydro, rt)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", flaky)
        win = gui_window.MainWindow(input_data)
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()

            win._on_gradients_clicked()
            _wait_until(
                lambda: win._worker is None
                and win._status_label.text().startswith("Gradients failed")
            )
            _pump()
            assert (
                win._status_label.text() == "Gradients failed: "
                "RuntimeError: boom"
            )
            assert win._gradient_state is None
            assert win._figure_panel.page_count() == 3
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            assert win._gradients_button.isEnabled()
            assert len(calls) == 2
            assert list((tmp_path / "results").glob("gradients_*.csv")) == []
        finally:
            _close_window(win)

    def test_no_free_parameters(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()

            def no_free() -> dict[str, tuple[float, float]]:
                return {
                    key: (value, value)
                    for key, value in win._param_panel.param_values().items()
                }

            monkeypatch.setattr(win._param_panel, "current_bounds", no_free)
            win._on_gradients_clicked()
            _pump()
            assert (
                win._status_label.text()
                == "No free parameters to differentiate."
            )
            assert win._worker is None
            assert len(calls) == 1
            assert win._gradients_button.isEnabled()
        finally:
            _close_window(win)


# ------------------------------------------------------------------ #
# The header's *Line search* button: the steepest-ascent jobs
# ------------------------------------------------------------------ #


class TestLineSearch:
    """The header's *Line search* button: the steepest-ascent jobs."""

    def test_menu_lists_finite_line_search_metrics(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The menu lists every finite objective with an optimizable metric."""
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        win = _gradient_window(
            input_data, tmp_path, monkeypatch, full_fake, hydro_fake
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()

            executed: list[QMenu] = []

            class _FakeLineSearchMenu(QMenu):
                """The window's menu: its exec is recorded, never opened."""

                # The PySide6 stubs misdeclare exec's return type as
                # QAction; Qt 6's QMenu::exec returns an int.
                def exec(self, *args: object) -> int:  # type: ignore[override]
                    executed.append(self)
                    return 0

            # A class-attribute patch cannot shadow the C-level method,
            # but the window resolves QMenu from its own module globals,
            # so the subclass stands in for the menu the click builds.
            monkeypatch.setattr(gui_window, "QMenu", _FakeLineSearchMenu)
            win._on_line_search_clicked()
            assert len(executed) == 1
            action_texts: list[str] = [
                action.text() for action in executed[0].actions()
            ]
            # Every finite cell of the table whose metric the search
            # can optimize: each target's four optimizable metrics,
            # in the table's column and row order.
            assert action_texts == [
                f"{target}: {metric}"
                for target in ("Streamflow", "DOC", "DIC", "Ca++")
                for metric in ("KGE", "LogKGE", "NSE", "LogNSE")
            ]
            # Showing the menu starts no job.
            assert win._worker is None
            assert win._line_search_state is None
        finally:
            _close_window(win)

    def test_full_mode_job(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A search with no gradient job computes its own direction."""
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        win = _gradient_window(
            input_data, tmp_path, monkeypatch, full_fake, hydro_fake
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            # The fake shifts the streamflow objectives by 2 * x of
            # free[0] alone, so the steepest ascent of Streamflow's
            # KGE is the unit direction along free[0]'s axis and the
            # best point is free[0]'s upper bound.
            mid: float = win._param_panel.param_values()[free[0]]
            high: float = win._param_panel.current_bounds()[free[0]][1]

            win._on_line_search_selected("Streamflow", "KGE", False)
            _wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            _pump()
            # The base point's results were handed in, the direction
            # came from its own central differences, then the 25
            # points were run.
            assert len(calls) == 1 + 2 * len(free) + 25

            assert win._line_search_state is not None
            profile: pd.DataFrame = win._line_search_state.profile
            assert len(profile) == 25
            assert profile.index.is_monotonic_increasing
            # The direction is unit length, so the farthest step is
            # the whole remaining distance to the upper bound.
            assert float(profile.index.max()) == pytest.approx(high - mid)

            # The best point is adopted into the panel: free[0] sits
            # at its upper bound, where the shifted value is
            # largest, and the results on screen are that point's,
            # so the save button lights up.
            assert (
                win._param_panel.param_values()[free[0]]
                == pytest.approx(high)
            )
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            assert win._gradients_button.isEnabled()
            assert win._line_search_button.isEnabled()

            # The profile is written once and rides along as the
            # figure panel's last page.
            csvs: list[Path] = list(
                (tmp_path / "results").glob("line_search_*.csv")
            )
            assert len(csvs) == 1
            assert win._line_search_state.csv_path is not None
            assert tmp_path / win._line_search_state.csv_path == csvs[0]
            assert win._figure_panel.page_count() == 4
            assert win._figure_panel._titles[-1] == "Line search (full)"

            assert re.fullmatch(
                r"Line search finished in \d+(?:\.\d+)? s: best t = [\d.]+ "
                r"\(Streamflow: KGE = [\d.]+\); the best point's parameters "
                r"are in the panel and the profile was saved to .*"
                r"line_search_\d{8}_\d{6}\.csv\.",
                win._status_label.text(),
            ) is not None
        finally:
            _close_window(win)

    def test_reuses_gradient_snapshot(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A finished gradient job's row supplies the direction for free."""
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        win = _gradient_window(
            input_data, tmp_path, monkeypatch, full_fake, hydro_fake
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            win._on_gradients_clicked()
            _wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            _pump()
            calls_after_gradient: int = len(calls)
            assert calls_after_gradient == 1 + 2 * len(free)

            mid: float = win._param_panel.param_values()[free[0]]
            high: float = win._param_panel.current_bounds()[free[0]][1]

            win._on_line_search_selected("Streamflow", "KGE", False)
            _wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            _pump()
            # No direction differences and no base re-run: the
            # snapshot's row supplies the direction and the base
            # point's results are handed in, so only the 25 points
            # are new calls.
            assert len(calls) == calls_after_gradient + 25

            assert win._line_search_state is not None
            profile: pd.DataFrame = win._line_search_state.profile
            assert len(profile) == 25
            # The snapshot's row is the raw difference, 2.0 along
            # free[0]'s axis, so the farthest step covers half the
            # remaining distance — yet the endpoint is the same
            # upper bound the unit direction reaches.
            assert float(profile.index.max()) == pytest.approx(
                (high - mid) / 2
            )
            assert (
                win._param_panel.param_values()[free[0]]
                == pytest.approx(high)
            )

            # Both jobs' pages ride along, in order.
            assert win._figure_panel.page_count() == 5
            assert win._figure_panel._titles[-2:] == [
                "Gradients (full)",
                "Line search (full)",
            ]
        finally:
            _close_window(win)

    def test_empty_objective_table(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no finished run, the click reports and starts nothing."""

        def failing(
            x: np.ndarray, _input_data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            raise ValueError("boom")

        win = _make_window(input_data, tmp_path, monkeypatch, failing)
        try:
            _wait_until(
                lambda: win._worker is None
                and win._status_label.text() == "Run failed: ValueError: boom"
            )
            _pump()
            win._on_line_search_clicked()
            assert (
                win._status_label.text()
                == "No objective values to search yet: run the model first."
            )
            assert win._worker is None
            assert win._line_search_state is None
        finally:
            _close_window(win)

    def test_no_free_parameters(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With every range collapsed, the selection is refused."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            def no_free() -> dict[str, tuple[float, float]]:
                return {
                    key: (value, value)
                    for key, value in win._param_panel.param_values().items()
                }

            monkeypatch.setattr(win._param_panel, "current_bounds", no_free)
            win._on_line_search_selected("Streamflow", "KGE", False)
            _pump()
            assert (
                win._status_label.text() == "No free parameters to search."
            )
            assert win._worker is None
            assert win._line_search_state is None
            assert len(calls) == 1
            assert win._line_search_button.isEnabled()
        finally:
            _close_window(win)
