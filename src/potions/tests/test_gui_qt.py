"""
Qt (PySide6) tests for the East River lateral-model calibration GUI.

These are the interactive-layer counterparts to `test_gui`, which
covers the data layer and the pure functions.  Here the actual
widgets are driven offscreen: the main window (startup, restoring a
saved run, background runs, saving, and closing with a run in
flight), the figure panel, the objective table model, and the
parameter panel (spin-box cells, pinned cells, their dialogs, and the
plot selection).

The offscreen Qt platform is requested before the first PySide6
import, and the model itself is never run: the window's
`arr_to_results` call is patched with a fake that returns the
synthetic results from `test_gui` (a real run takes about five
minutes and is covered by that file's own slow test).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QAction, QCloseEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QVBoxLayout,
    QWidget,
)

import potions as pt

from ..gui import data as gui_data
from ..gui import figure_panel as gui_figure_panel
from ..gui import param_panel as gui_param_panel
from ..gui import window as gui_window
from ..gui.common import (
    HYDRO_THEME,
    METRIC_LABELS,
    SPECIATION_ZONES,
    param_label,
)
from ..gui.data import PlotSelection
from ..gui.state import read_gui_state
from .test_gui import (
    _full_state,
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
) -> Callable[[np.ndarray, gui_data.GuiInputData], pt.ModelResults]:
    """
    A stand-in for `arr_to_results` that records each parameter
    vector it is called with and returns the synthetic results.
    """

    def fake(
        x: np.ndarray, _input_data: gui_data.GuiInputData
    ) -> pt.ModelResults:
        calls.append(np.array(x))
        return pt.ModelResults(*_synthetic_results(input_data))

    return fake


def _make_window(
    input_data: gui_data.GuiInputData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: Callable[[np.ndarray, gui_data.GuiInputData], pt.ModelResults],
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
    results: pt.ModelResults | None = win._results
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

    Each constructed dialog is recorded (title, prefilled ends, and
    hint) and `run()` returns `answer` (None stands in for a
    cancelled dialog).

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
            parent: object | None = None,
        ) -> None:
            records.append({"title": title, "lo": lo, "hi": hi, "hint": hint})

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


class _FakeMessageBox:
    """
    Stands in for `QMessageBox`; the window only calls `question`,
    which returns the class-level `answer`.
    """

    class StandardButton:
        No: int = 0
        Yes: int = 1

    answer: int = StandardButton.No

    @staticmethod
    def question(
        parent: object | None,
        title: str,
        text: str,
        buttons: int = 0,
    ) -> int:
        return _FakeMessageBox.answer


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
            assert win._progress_bar.isHidden()
            assert win._elapsed_label.isHidden()
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            assert panel.page_count() == 3
            assert panel.current_page() == 0
            assert panel._titles == [
                "Streamflow and river concentrations",
                "Zone concentrations and storage",
                "Mineral reaction rates",
            ]
            assert win._objective_model.rowCount() == 4
            assert win._objective_model.columnCount() == 7
            assert win.windowTitle() == gui_window._TITLE
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
            tmp_path, input_data, _full_state(input_data)
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
            panel: gui_param_panel.ParamPanel = win._param_panel
            assert panel.param_values()["proportion.1"] == pytest.approx(0.7)
            assert panel.current_bounds() == dict(input_data.bounds)
            assert panel.selection() == gui_data.default_plot_selection()
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
            # The requested sizes are scaled to the splitter's actual
            # height, with each section clamped to its child's minimum
            # size hint: the proportions survive up to that clamping,
            # so a loose tolerance is the right assertion.
            sizes: list[float] = [float(s) for s in win._splitter.sizes()]
            total: float = sum(sizes)
            assert total > 0
            for size, target in zip(sizes, (300, 200, 400)):
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
        ) -> pt.ModelResults:
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
            assert win._figure_panel.page_count() == 0
            assert win._objective_model.rowCount() == 0
            win._on_save_run()
            assert win._status_label.text() == (
                "Nothing to save yet: the model has not run."
            )
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
                run_dir.mkdir(parents=True)
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
            assert win._status_label.text() == f"Saved run to {run_dir}."

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
            assert saved.splitter_sizes is not None
            sizes: list[float] = [float(s) for s in saved.splitter_sizes]
            total: float = sum(sizes)
            assert total > 0
            initial_total: int = sum(gui_window._INITIAL_SPLIT_SIZES)
            for size, target in zip(sizes, gui_window._INITIAL_SPLIT_SIZES):
                assert size / total == pytest.approx(
                    target / initial_total, abs=0.02
                )

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
            assert len(saves) == 1
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
        ) -> pt.ModelResults:
            release: threading.Event = threading.Event()
            releases.append(release)
            release.wait()
            return pt.ModelResults(*_synthetic_results(_input_data))

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


# ------------------------------------------------------------------ #
# _ObjectiveTableModel (standalone)
# ------------------------------------------------------------------ #


def _objective_table() -> pd.DataFrame:
    """A two-row objective table with NaNs in the second row."""
    values: dict[str, list[float]] = {
        "kge": [0.9, float("nan")],
        "log_kge": [0.5, float("nan")],
        "nse": [0.8, float("nan")],
        "log_nse": [0.4, float("nan")],
        "bias": [0.01, float("nan")],
        "spearman_rho": [0.7, float("nan")],
        "r_squared": [0.6, float("nan")],
    }
    return pd.DataFrame(values, index=["Streamflow", "DOC"])


class TestObjectiveTableModel:
    """The model behind the table strip under the figure."""

    def test_empty_model(self, qapp: QApplication) -> None:
        model = gui_window._ObjectiveTableModel()
        try:
            assert model.rowCount() == 0
            assert model.columnCount() == 7
            headers = [
                model.headerData(i, Qt.Orientation.Horizontal)
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
            assert model.rowCount() == 2
            row_headers = [
                model.headerData(i, Qt.Orientation.Vertical) for i in range(2)
            ]
            assert row_headers == ["Streamflow", "DOC"]
            assert model.data(model.index(0, 0)) == "0.9000"
            assert model.data(model.index(1, 0)) == "–"
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
            qapp.sendEvent(row._spin, _double_click_event())
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

            # set_selection updates the boxes without emitting and
            # ignores names the network does not know.
            panel.set_selection(
                PlotSelection(
                    species=["DOC", "Bogus"],
                    zones=SPECIATION_ZONES,
                    minerals=["SOC(s)"],
                )
            )
            assert len(selections) == 1
            assert panel.selection().species == ["DOC"]
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
            assert isinstance(panel._plots_group._content_layout, QVBoxLayout)

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
