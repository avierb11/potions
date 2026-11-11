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
import, and the model itself is never run: `arr_to_results` and
`arr_to_hydro_results` are patched with fakes that return the
synthetic results from `test_gui` (a real run takes about five
minutes and is covered by that file's own slow test).  The
in-process model run resolves the fakes from the `window`
module; the gradient and line-search jobs resolve theirs from
the `data` module, because their points run in forked pool
workers that inherit the patched module attributes (the pool
initializers still load the small packaged input CSV; the
fakes ignore its contents).
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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text
from PySide6.QtCore import QDate, QEvent, QPointF, QRect, QSize, Qt
from PySide6.QtGui import QAction, QCloseEvent, QFont, QImage, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayoutItem,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

import potions as pt
from potions.hydro_model import HydrologicalModel
from potions.reactive_transport.common_networks import (
    get_carbon_network,
    get_soc_calcite_network,
)
from potions.reactive_transport.rt_zone import WaterVolumeError

from ..gui import data as gui_data
from ..gui import calibration as gui_calibration
from ..gui import figure_panel as gui_figure_panel
from ..gui import model_specs
from ..gui import network_editor as gui_network_editor
from ..gui import network_specs
from ..gui import param_panel as gui_param_panel
from ..gui import window as gui_window
from ..gui.common import (
    HYDRO_THEME,
    METRIC_LABELS,
    METRIC_NAMES,
    RIVER_SPECIES,
    RT_THEME,
    SPECIATION_ZONES,
    param_label,
)
from ..gui.data import PlotSelection
from ..gui.gradients import (
    _Perturbation,
    free_parameter_steps,
    gradient_matrix,
    objective_vector,
)
from ..gui.line_search import expand_direction, feasible_t_interval
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
    model_inputs,
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
        hydro, rt = _synthetic_results(_input_data)
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


def _app_font_12pt() -> QFont:
    """The app's 12 pt base font, as `main` sets it on the app.

    The test windows never pass through `main`, so the tests that
    compare pixel-level rendering set the font on the widgets they
    probe.
    """
    font: QFont = QFont()
    font.setPointSize(12)
    return font


def _objective_test_table() -> pd.DataFrame:
    """A four-target objective table crafted for the panel tests.

    The three species' values format to eight-character scientific
    strings (``"1.23e+03"`` and the like) that are wider, at the
    app's 12 pt font, than their short headers (``"DOC"``, ``"DIC"``,
    ``"Ca++"``), so each of those rendered columns is wider than a
    header-text-only formula; the streamflow values format to short
    strings (``"0.9"`` through ``"6.3"``) that stay within their
    longer ``"Streamflow"`` header, so that column is not.
    """
    values: dict[str, list[float]] = {
        "Streamflow": [0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3],
        "DOC": [1.23e3, 2.34e3, 3.45e3, 4.56e3, 5.67e3, 6.78e3, 7.89e3],
        "DIC": [1.23e4, 2.34e4, 3.45e4, 4.56e4, 5.67e4, 6.78e4, 7.89e4],
        "Ca++": [1.23e5, 2.34e5, 3.45e5, 4.56e5, 5.67e5, 6.78e5, 7.89e5],
    }
    return pd.DataFrame(values, index=list(METRIC_LABELS.values()))


class _FakeMessageBox:
    """
    Stands in for `QMessageBox`; the window calls `question`
    (which returns the class-level `answer`), `critical`, and
    `information` (whose calls are recorded).
    """

    class StandardButton:
        No: int = 0
        Yes: int = 1

    answer: int = StandardButton.No
    critical_calls: list[tuple[str, str]] = []
    information_calls: list[tuple[str, str]] = []

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

    @staticmethod
    def information(
        parent: object | None,
        title: str,
        text: str,
        *buttons: object,
    ) -> None:
        _FakeMessageBox.information_calls.append((title, text))


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
            assert panel.page_count() == 8
            assert panel.current_page() == 0
            assert panel._titles == [
                "Hydrology",
                "Storage",
                "Partitioning",
                "River concentrations",
                "River loads",
                "Concentration vs. discharge",
                "Zone concentrations",
                "Reaction rates (SOC(s))",
            ]
            # The table is transposed: one row per metric, one column
            # per measured target (the streamflow and the species).
            assert win._objective_model.rowCount() == 7
            assert win._objective_model.columnCount() == 4
            assert win.windowTitle() == gui_window._title_for(
                model_specs.DEFAULT_MODEL_SPEC
            )
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
            # own measurements demand, per the pure sizing function:
            # the rendered section sizes (both headers re-resized
            # to contents, the wider of the header text and the
            # cell content), the headers' own size hints, and the
            # frame width.
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
            table = win._objective_table
            table.horizontalHeader().resizeSections()
            table.verticalHeader().resizeSections()
            width, height = gui_window._objective_panel_size(
                win._objective_heading.sizeHint(),
                [
                    table.horizontalHeader().sectionSize(c)
                    for c in range(win._objective_model.columnCount())
                ],
                [
                    table.verticalHeader().sectionSize(r)
                    for r in range(win._objective_model.rowCount())
                ],
                table.verticalHeader().sizeHint().width(),
                table.horizontalHeader().sizeHint().height(),
                table.frameWidth(),
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

    def test_objective_panel_width_uses_rendered_section_sizes(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The panel's fixed width fits the rendered column sections.

        The width formula must use the section sizes the view
        actually renders (the wider of the header text and the cell
        content), not the header-text-only size hints: at the app's
        12 pt font the value cells are wider than the short species
        headers, and a header-only formula comes out narrow enough
        to clip the last column.
        """
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
            font: QFont = _app_font_12pt()
            win._objective_table.setFont(font)
            win._objective_heading.setFont(font)
            win._objective_model.set_table(_objective_test_table())
            win._fit_objective_panel()
            table = win._objective_table
            h: QHeaderView = table.horizontalHeader()
            v: QHeaderView = table.verticalHeader()
            h.resizeSections()
            v.resizeSections()
            # The crafted table discriminates: some rendered column is
            # wider than the header-text-only formula.
            assert any(
                h.sectionSize(c)
                > h.sectionSizeHint(c) + gui_window._CHECKBOX_INSET
                for c in range(win._objective_model.columnCount())
            )
            # Every rendered column fits its widest cell.
            for c in range(win._objective_model.columnCount()):
                widest: int = 0
                for r in range(win._objective_model.rowCount()):
                    hint: QSize = table.itemDelegate().sizeHint(
                        QStyleOptionViewItem(),
                        win._objective_model.index(r, c),
                    )
                    widest = max(widest, hint.width())
                assert h.sectionSize(c) >= widest
            width, height = gui_window._objective_panel_size(
                win._objective_heading.sizeHint(),
                [
                    h.sectionSize(c)
                    for c in range(win._objective_model.columnCount())
                ],
                [
                    v.sectionSize(r)
                    for r in range(win._objective_model.rowCount())
                ],
                v.sizeHint().width(),
                h.sizeHint().height(),
                table.frameWidth(),
                gui_window._OBJECTIVE_PANEL_MARGINS,
                gui_window._OBJECTIVE_PANEL_SPACING,
            )
            assert (
                win._objective_panel.width(),
                win._objective_panel.height(),
            ) == (width, height)
        finally:
            _close_window(win)

    def test_objective_cell_value_clears_the_checkbox_inset(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cell's value text sits clear of the checkbox, fully drawn.

        The delegate paints an opaque checkbox over each cell's left
        inset, so the value text must be shifted past it: no glyph
        may start inside the inset, and the whole text must be
        visible, ending near the inset plus the style's text margin
        plus the font-metrics width (a covered start or a clipped
        end would shorten the visible span).
        """
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
            font: QFont = _app_font_12pt()
            win._objective_table.setFont(font)
            win._objective_heading.setFont(font)
            win._objective_model.set_table(_objective_test_table())
            win._fit_objective_panel()
            table = win._objective_table
            h: QHeaderView = table.horizontalHeader()
            v: QHeaderView = table.verticalHeader()
            h.resizeSections()
            v.resizeSections()
            # Lay the table out explicitly (the test window is never
            # shown): give it its full content size so the headers
            # and viewport are positioned, then grab the viewport —
            # `visualRect` is in viewport coordinates, so no parent
            # mapping is involved.
            table.resize(
                v.sizeHint().width()
                + sum(
                    h.sectionSize(c)
                    for c in range(win._objective_model.columnCount())
                )
                + 2 * table.frameWidth(),
                h.sizeHint().height()
                + sum(
                    v.sectionSize(r)
                    for r in range(win._objective_model.rowCount())
                )
                + 2 * table.frameWidth(),
            )
            for _ in range(20):
                qapp.processEvents()
                time.sleep(0.002)
            assert table.viewport().width() > 0
            assert table.viewport().height() > 0
            model = win._objective_model
            idx = model.index(1, model.columnCount() - 1)
            cell = table.visualRect(idx)
            assert cell.width() > 0 and cell.height() > 0
            text: str | None = model.data(idx, Qt.ItemDataRole.DisplayRole)
            assert isinstance(text, str)
            tw: int = table.fontMetrics().horizontalAdvance(text)
            # The probed text is wide enough that the defect (a
            # quarter of it hidden under the 24 px checkbox) is
            # resolvable at the pixel level.
            assert tw >= 30
            pixmap = table.viewport().grab()
            img: QImage = pixmap.toImage()
            s_x: float = pixmap.width() / table.viewport().width()
            s_y: float = pixmap.height() / table.viewport().height()

            def local_dark_xs() -> list[float]:
                dark: list[float] = []
                for yv in range(cell.y(), cell.y() + cell.height()):
                    gy: int = int(round(yv * s_y))
                    if gy < 0 or gy >= img.height():
                        continue
                    for xv in range(cell.x(), cell.x() + cell.width()):
                        gx: int = int(round(xv * s_x))
                        if gx < 0 or gx >= img.width():
                            continue
                        if (img.pixel(gx, gy) & 0xFF) < 128:
                            dark.append(xv - cell.x())
                return dark

            dark = local_dark_xs()
            assert dark, "the cell drew no dark pixels at all"
            first_dark: float = min(dark)
            last_dark: float = max(dark)
            # The text starts to the right of the checkbox's inset
            # (a little slack for antialiasing): it is not covered
            # at its beginning.
            assert first_dark >= gui_window._CHECKBOX_INSET - 2
            # The full text is visible: the ink span matches the
            # font-metrics width of the display string.
            assert abs((last_dark - first_dark) - tw) <= 8
            # The text ends where a shifted text should end: inset
            # plus the style's 2 px text margin plus the full width
            # (the ink trail ends a pixel or two short of the
            # advance width): it is not clipped at the cell edge.
            assert abs(last_dark - (gui_window._CHECKBOX_INSET + 2 + tw)) <= 10
            # Guard: the checkbox itself is still drawn in the
            # inset: its white fill (well above the cell background)
            # at the cell's vertical middle.
            mid_y: int = int(round((cell.y() + cell.height() // 2) * s_y))
            fill: bool = any(
                img.pixel(int(round((cell.x() + lx) * s_x)), mid_y) & 0xFF
                >= 250
                for lx in range(3, 12)
            )
            assert fill, "the checkbox's white fill is missing from the inset"
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
            assert win._figure_panel.page_count() == 8
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
            assert win.width() == gui_window._INITIAL_SIZE[0]
            assert win.height() == gui_window._INITIAL_SIZE[1]
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
            # The three hydrology figures, one target column.
            figure: gui_figure_panel.FigurePanel = win._figure_panel
            assert figure.page_count() == 3
            assert figure._titles == [
                "Hydrology",
                "Storage",
                "Partitioning",
            ]
            assert figure._title_label.text() == "Hydrology"
            assert figure._page_label.text() == "1 / 3"
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
            # is exactly the geometry the constructor asked for.  The
            # header's minimum width can exceed the initial width, so
            # record the geometry Qt actually applied.
            win.setGeometry(0, 0, *gui_window._INITIAL_SIZE)
            _pump()
            applied: QRect = win.geometry()
            expected_geometry: dict[str, int] = {
                "x": applied.x(),
                "y": applied.y(),
                "width": applied.width(),
                "height": applied.height(),
            }
            monkeypatch.setattr(gui_window, "save_model_run", fake_save)
            win._on_save_run()
            assert len(saves) == 1
            assert saves[0]["results_dir"] is None
            assert win._status_label.text() == (
                f"Saved run to {run_dir}; 8 figure page(s) saved to "
                f"{run_dir / 'figures'}."
            )
            # The figures are saved as plotted, named after their
            # page titles (non-alphanumerics collapse to a single
            # underscore).
            figure_names: list[str] = sorted(
                path.name for path in (run_dir / "figures").glob("*.png")
            )
            assert figure_names == [
                "Concentration_vs_discharge.png",
                "Hydrology.png",
                "Partitioning.png",
                "Reaction_rates_SOC_s.png",
                "River_concentrations.png",
                "River_loads.png",
                "Storage.png",
                "Zone_concentrations.png",
            ]

            saved = read_gui_state(run_dir)
            assert saved is not None
            assert saved.csv_path is None
            assert saved.model_key == "lateral_three_layer"
            assert saved.param_values == win._param_panel.param_values()
            assert saved.param_bounds == win._param_panel.current_bounds()
            assert saved.selection == gui_data.default_plot_selection()
            # The window was shown and restored to the default origin
            # and size, so it saves exactly that client geometry; the
            # splitter saves its laid-out sizes, i.e. the initial
            # distribution scaled to the window's height.  Compare
            # proportions rather than raw pixels.
            assert saved.window_geometry == expected_geometry
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
            assert panel._title_label.text() == "Storage"
            assert panel._page_label.text() == "2 / 8"
            actions["Next figure"].triggered.emit()
            assert panel.current_page() == 2
            assert panel._title_label.text() == "Partitioning"
            assert panel._page_label.text() == "3 / 8"
            # Wrap-around in both directions.
            for _ in range(6):
                actions["Next figure"].triggered.emit()
            assert panel.current_page() == 0
            assert panel._page_label.text() == "1 / 8"
            actions["Previous figure"].triggered.emit()
            assert panel.current_page() == 7
            assert panel._title_label.text() == "Reaction rates (SOC(s))"
            assert panel._page_label.text() == "8 / 8"
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
            assert win._figure_panel.page_count() == 8
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
            assert win._figure_panel.page_count() == 3
            assert win._figure_panel._titles == [
                "Hydrology",
                "Storage",
                "Partitioning",
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
                f"Saved run to {run_dir}; 3 figure page(s) saved to "
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
            assert win._figure_panel.page_count() == 8
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
# MainWindow: mode tint
# ------------------------------------------------------------------ #


class TestModeTint:
    """The window's background tint by run mode."""

    def test_mode_background_tint(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Toggling the mode checkbox tints the window background."""
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
            # The startup run finished in the default full mode, so
            # the window carries the warm reactive-transport tint.
            assert win.styleSheet() == "background-color: #FBEEDB;"
            # With auto-run off, toggling the mode updates the panel
            # and the tint but starts no re-run.
            win._auto_run_checkbox.setChecked(False)
            win._mode_checkbox.setChecked(True)
            assert win.styleSheet() == "background-color: #EAF1F8;"
            win._mode_checkbox.setChecked(False)
            assert win.styleSheet() == "background-color: #FBEEDB;"
            assert len(calls) == 1
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
            assert win._figure_panel.page_count() == 8
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
            assert win._figure_panel.page_count() == 3
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

            # Page 0 is the hydrology figure: the previous run's
            # line sits behind the current one, half transparent.
            figure: Figure = win._figure_panel._canvases[0].figure
            hydro_lines = _lines(figure.axes[0])
            assert len(hydro_lines) == 2
            assert hydro_lines[0].get_alpha() == 0.5
            assert _alpha(hydro_lines[1]) == 1.0
            # Both lines draw in the simulated-discharge color; the
            # alpha is what sets the previous run's apart.
            assert np.array_equal(
                hydro_lines[0].get_color(), hydro_lines[1].get_color()
            )
            # The legend still lists only the current run's series.
            assert [t.get_text() for t in figure.legends[0].get_texts()] == [
                "Streamflow (sim)",
                "Streamflow (meas)",
            ]

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

    def test_hidpi_scaling_uses_the_canvas_dpi(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The display's device pixel ratio rescales a figure from the
        canvas dpi, not the publication dpi.

        Under the package's rcParams a figure is stamped with a
        300 dpi when it is constructed, before the panel restates it
        at `CANVAS_DPI`.  A canvas that rescaled from that stale
        stamp would render at the ratio times 300 (three times too
        large on a ratio of 4/3) instead of times `CANVAS_DPI`.
        """
        # `main` sets these rcParams before building the figures;
        # the panel's tests bypass it, so the tests set them here.
        monkeypatch.setitem(rcParams, "figure.dpi", 300)
        monkeypatch.setitem(rcParams, "font.size", 12)
        dpr: float = 4 / 3
        panel = gui_figure_panel.FigurePanel()
        try:
            panel.set_figures([_figure("A"), _figure("B")])
            for canvas in panel._canvases:
                canvas._set_device_pixel_ratio(dpr)
            for canvas in panel._canvases:
                assert canvas.figure.dpi == pytest.approx(
                    gui_figure_panel.CANVAS_DPI * dpr, rel=1e-6
                )
        finally:
            panel.deleteLater()
            _pump()

    def test_hidpi_legend_text_stays_at_canvas_scale(
        self, qapp: QApplication, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A legend's text renders at the base font of the canvas dpi,
        not the publication dpi, on a high-dpi display.

        At a device pixel ratio of 4/3 the figure's dpi is the
        canvas dpi times the ratio, so the 12-point legend text
        occupies a bounded number of physical pixels; a figure
        rescaled from the 300 dpi publication stamp occupies three
        times as many and the legend covers the plot.
        """
        monkeypatch.setitem(rcParams, "figure.dpi", 300)
        monkeypatch.setitem(rcParams, "font.size", 12)
        dpr: float = 4 / 3
        panel = gui_figure_panel.FigurePanel()
        try:
            fig: Figure = Figure(figsize=(15, 7))
            ax = fig.add_subplot()
            ax.plot([0.0, 1.0], [0.0, 1.0], label="Series")
            leg: Legend = fig.legend()
            panel.set_figures([("Streamflow", fig)])

            canvas = panel._canvases[0]
            canvas._set_device_pixel_ratio(dpr)
            renderer = canvas.get_renderer()  # type: ignore[no-untyped-call]
            # One em of the 12-point base font at the rescaled dpi.
            em: float = 12 / 72 * gui_figure_panel.CANVAS_DPI * dpr
            texts: list[Text] = leg.get_texts()
            assert texts, "Expected the legend to hold text."
            max_height: float = max(
                text.get_window_extent(renderer).height for text in texts
            )
            # Measured line heights of 12-point text sit at or below
            # one em at the figure dpi, so the 1.5x margin bounds any
            # font metrics while still rejecting the three-times
            # larger publication-dpi height.
            assert max_height <= 1.5 * em
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
            assert model.data(model.index(0, 0)) == "0.9"
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

    def test_cell_formats_three_visible_digits(
        self, qapp: QApplication
    ) -> None:
        """A cell shows three visible digits; scientific when large."""
        values: dict[str, list[float]] = {
            "Streamflow": [0.9, 0.876, 1234.5, 0.0000123, 0.0009, 99.9, 100.0],
        }
        model = gui_window._ObjectiveTableModel()
        try:
            model.set_table(
                pd.DataFrame(values, index=list(METRIC_LABELS.values()))
            )
            texts: list[str | None] = [
                model.data(model.index(row, 0)) for row in range(7)
            ]
            # Three significant digits, so 1234.5 needs scientific
            # notation and 0.0009 shows its single nonzero digit
            # instead of pre-rounding to zero.
            assert texts == [
                "0.9",
                "0.876",
                "1.23e+03",
                "1.23e-05",
                "0.0009",
                "99.9",
                "100",
            ]
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
            box: QCheckBox = panel._streamflow_box
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

    def test_symbol_double_click_requests_edit(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Double-clicking the symbol label opens the range dialog too:
        the label, like the line edit, swallows the click before it
        can reach the cell's double-click handler, so it is watched
        for one as well.
        """
        panel = self._make_panel(input_data)
        try:
            key: str = "proportion.1"
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel._FreeCell)

            records = _install_bounds_stub(monkeypatch, (0.5, 0.95))
            qapp.sendEvent(row._symbol, _double_click_event())
            assert len(records) == 1
            assert records[0]["lo"] == pytest.approx(0.25)
            assert records[0]["hi"] == pytest.approx(0.95)
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
        """Toggling a control emits the recomputed selection; set_selection is silent."""
        panel = self._make_panel(input_data)
        try:
            selections: list[PlotSelection] = []
            panel.selection_changed.connect(selections.append)
            panel._species_boxes["DIC"].setChecked(False)
            assert len(selections) == 1
            selection = selections[0]
            assert selection.species == ["DOC", "Ca++"]
            assert selection.zones == SPECIATION_ZONES
            assert selection.mineral == "SOC(s)"
            assert selection.streamflow is True

            # set_selection updates the controls without emitting
            # and ignores names the panel does not offer.
            panel.set_selection(
                PlotSelection(
                    species=["DOC", "Bogus"],
                    zones=SPECIATION_ZONES,
                    mineral="Calcite(s)",
                    streamflow=False,
                    window_start="2016-01-01",
                    window_end="2020-12-31",
                )
            )
            assert len(selections) == 1
            assert panel.selection().species == ["DOC"]
            assert panel.selection().mineral == "Calcite(s)"
            assert panel.selection().streamflow is False
            assert panel.selection().window_start == "2016-01-01"
            assert panel.selection().window_end == "2020-12-31"
        finally:
            panel.deleteLater()
            _pump()

    def test_species_boxes_follow_measured_columns(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        """The species checkboxes follow the run's measured columns."""
        panel = self._make_panel(input_data)
        try:
            assert list(panel._species_boxes) == list(input_data.conc_df.columns)
            assert panel.selection().species == ["DOC", "Ca++", "DIC"]
        finally:
            panel.deleteLater()
            _pump()

    def test_plotting_date_window(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        """Moving the date pickers keeps from below to and re-emits."""
        panel = self._make_panel(input_data)
        try:
            selections: list[PlotSelection] = []
            panel.selection_changed.connect(selections.append)
            assert [
                box.isChecked() for box in panel._species_boxes.values()
            ] == [True, True, True]

            # Lower the to date; the from date (the record's start)
            # stays put.
            panel._window_to_edit.setDate(QDate(2019, 9, 30))
            assert len(selections) == 1
            assert selections[0].window_start == gui_data.DEFAULT_WINDOW_START
            assert selections[0].window_end == "2019-09-30"

            # Raise the from date past the to date: the to date
            # lifts to it.
            panel._window_from_edit.setDate(QDate(2020, 1, 1))
            assert len(selections) == 2
            assert selections[1].window_start == "2020-01-01"
            assert selections[1].window_end == "2020-01-01"

            # Lower the to date below the from date: the from date
            # lowers to it.
            panel._window_to_edit.setDate(QDate(2016, 1, 1))
            assert len(selections) == 3
            assert selections[2].window_start == "2016-01-01"
            assert selections[2].window_end == "2016-01-01"
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

    def test_panel_uses_spec_zone_labels(
        self, qapp: QApplication
    ) -> None:
        data = gui_data.load_input_data(
            gui_data.DEFAULT_CSV_PATH, model_key="hbv"
        )
        panel = gui_param_panel.ParamPanel(
            data, gui_data.default_plot_selection(data.spec)
        )
        try:
            labels = [
                header.text()
                for group in panel._groups.values()
                for header in group.findChildren(
                    gui_param_panel._ZoneHeader
                )
            ]
            assert any("Shallow subsurface" in text for text in labels)
            assert any("Deep subsurface" in text for text in labels)
            assert panel._groups.get("proportion") is None
        finally:
            panel.deleteLater()
            _pump()

    def test_reload_switches_groups_and_defaults(
        self, qapp: QApplication
    ) -> None:
        data = gui_data.load_input_data(
            gui_data.DEFAULT_CSV_PATH, model_key="lateral_three_layer"
        )
        panel = gui_param_panel.ParamPanel(
            data, gui_data.default_plot_selection(data.spec)
        )
        try:
            old_keys = set(panel.param_values())
            hbv = gui_data.load_input_data(
                gui_data.DEFAULT_CSV_PATH, model_key="hbv"
            )
            panel.reload(hbv, gui_data.default_plot_selection(hbv.spec))
            new_keys = set(panel.param_values())
            assert "shallow.perc" in new_keys
            assert "snow_hs.tt" not in new_keys
            assert old_keys != new_keys
            assert set(panel._zone_boxes) == set(hbv.spec.zones)
            assert panel._groups.get("proportion") is None
        finally:
            panel.deleteLater()
            _pump()

    def test_reload_preserves_hydro_only_visibility(
        self, qapp: QApplication
    ) -> None:
        data = gui_data.load_input_data(
            gui_data.DEFAULT_CSV_PATH, model_key="hbv"
        )
        panel = gui_param_panel.ParamPanel(
            data, gui_data.default_plot_selection(data.spec)
        )
        try:
            panel.set_hydro_only(True)
            panel.reload(data, gui_data.default_plot_selection(data.spec))
            assert not panel._groups["reactive"].isVisible()
            assert not panel._groups["initial"].isVisible()
        finally:
            panel.deleteLater()
            _pump()


# ------------------------------------------------------------------ #
# The range dialog's own buttons (standalone)
# ------------------------------------------------------------------ #


class TestBoundsDialog:
    """The range dialog's own buttons: OK accepts, Cancel rejects."""

    def test_ok_accepts_and_cancel_rejects(self, qapp: QApplication) -> None:
        """
        Clicking OK accepts the dialog and clicking Cancel rejects
        it, so a modal run returns Accepted after the former and a
        cancelled result after the latter; the button box is wired to
        the dialog explicitly, since it does not do so itself.
        """
        dialog = gui_param_panel._BoundsDialog(
            "Edit range — Test parameter",
            0.25,
            0.95,
            "Current range: 0.25000 to 0.95000.",
            0.6,
        )
        try:
            events: list[str] = []
            dialog.accepted.connect(lambda: events.append("accepted"))
            dialog.rejected.connect(lambda: events.append("rejected"))

            dialog._ok_button.click()
            assert dialog.result() == QDialog.DialogCode.Accepted
            assert events == ["accepted"]

            box: QDialogButtonBox | None = dialog.findChild(
                QDialogButtonBox
            )
            assert box is not None
            cancel: QPushButton = box.button(
                QDialogButtonBox.StandardButton.Cancel
            )
            cancel.click()
            assert dialog.result() == QDialog.DialogCode.Rejected
            assert events == ["accepted", "rejected"]
        finally:
            dialog.deleteLater()
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
        hydro, rt = _synthetic_results(_input_data)
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


def _file_gated_fake(
    real_fake: Callable[
        [np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults
    ],
    gate_path: Path,
) -> Callable[[np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults]:
    """
    Wrap `real_fake` in a gate the test releases by touching
    `gate_path`: each call blocks (polling briefly, with a
    generous timeout) until the file exists, then delegates.

    Unlike a `threading.Event`, a file is visible to the forked
    pool worker processes the jobs use, so the test can hold a
    job's points until it has connected to the worker's signals.
    """

    def gated_fake(
        x: np.ndarray, _input_data: gui_data.GuiInputData
    ) -> gui_data.GuiRunResults:
        deadline: float = time.monotonic() + 60.0
        while not gate_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return real_fake(x, _input_data)

    return gated_fake


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
    gate_path: Path | None = None,
) -> gui_window.MainWindow:
    """
    Build a main window in an empty `tmp_path` with `full_fake` and
    `hydro_fake` patched in for `arr_to_results` and
    `arr_to_hydro_results`, both on the `window` module (the
    in-process startup run) and on the `data` module (the forked
    pool workers that run the jobs' points).  When `gate_path` is
    given, the `data` copies block until the file is touched, so a
    job's test can connect to its worker's signals first.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gui_window, "arr_to_results", full_fake)
    monkeypatch.setattr(gui_window, "arr_to_hydro_results", hydro_fake)
    if gate_path is None:
        monkeypatch.setattr(gui_data, "arr_to_results", full_fake)
        monkeypatch.setattr(gui_data, "arr_to_hydro_results", hydro_fake)
    else:
        monkeypatch.setattr(
            gui_data,
            "arr_to_results",
            _file_gated_fake(full_fake, gate_path),
        )
        monkeypatch.setattr(
            gui_data,
            "arr_to_hydro_results",
            _file_gated_fake(hydro_fake, gate_path),
        )
    return gui_window.MainWindow(input_data)


def _only_csv(tmp_path: Path) -> Path:
    """The single ``gradients_*.csv`` in `tmp_path`'s results directory."""
    csvs: list[Path] = list((tmp_path / "results").glob("gradients_*.csv"))
    assert len(csvs) == 1
    return csvs[0]


# ------------------------------------------------------------------ #
# _SpinnerLabel (standalone)
# ------------------------------------------------------------------ #


class TestSpinnerLabel:
    """The header's running-job indicator."""

    def test_spinner_label_cycles_frames(self, qapp: QApplication) -> None:
        """
        The indicator is hidden at rest and keeps a fixed 24 px
        footprint; while spinning it shows and cycles its frames on
        each timer tick, and stopping hides it again and clears the
        frame.
        """
        label = gui_window._SpinnerLabel()
        try:
            # Hidden at rest, with the fixed square footprint.
            assert label.isHidden()
            assert label.text() == ""
            assert label.minimumWidth() == gui_window._SPINNER_SIZE
            assert label.maximumWidth() == gui_window._SPINNER_SIZE
            assert label.minimumHeight() == gui_window._SPINNER_SIZE
            assert label.maximumHeight() == gui_window._SPINNER_SIZE

            # Spinning shows the current frame and advances it on
            # each tick; the ~200 ms pump below spans at least two
            # 100 ms periods, but the exact tick count depends on
            # timer drift, so only "changed" is asserted.
            label.set_spinning(True)
            assert not label.isHidden()
            first_frame: str = label.text()
            assert first_frame in gui_window._SPINNER_FRAMES
            _pump(gui_window._SPINNER_INTERVAL_MS + 50)
            assert label.text() != first_frame
            assert label.text() in gui_window._SPINNER_FRAMES

            # Stopping hides the indicator and clears the frame.
            label.set_spinning(False)
            assert label.isHidden()
            assert label.text() == ""
        finally:
            label.deleteLater()
            _pump()


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
        gate = tmp_path / "job_gate"
        win = _gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
            gate_path=gate,
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window._GradientWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            _wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            _pump()
            # The job's points ran in the pool (their `calls`
            # appends never reach this process): only the
            # in-process startup run is recorded, and the last
            # step completes the bar.
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) <= total <= 2 * len(free)

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

            assert win._figure_panel.page_count() == 9
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
        gate = tmp_path / "job_gate"
        win = _gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
            gate_path=gate,
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
            worker = win._worker
            assert isinstance(worker, gui_window._GradientWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            _wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            _pump()
            # The job's points ran in the pool (their `calls`
            # appends never reach this process): the two in-process
            # runs (the startup and the mode change) are the only
            # calls recorded, and the last step completes the bar.
            assert len(calls) == 2
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) <= total <= 2 * len(free)

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

            assert win._figure_panel.page_count() == 4
            assert win._figure_panel._titles[-1] == "Gradients (hydro)"
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            assert win._gradients_button.isEnabled()
        finally:
            _close_window(win)

    def test_gradient_job_failed_point_becomes_nan(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A pool point that fails does not fail the job: its column
        of the difference matrix is all NaN (the missing value),
        and the completion status counts the failure.
        """
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        plain_fake = _fake_results_factory(calls, input_data)
        failing_index: int = list(input_data.bounds).index(free[0])

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", plain_fake)
        win = gui_window.MainWindow(input_data)
        gate = tmp_path / "job_gate"
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()

            # Only the free[0] "+" perturbation exceeds the base
            # value: the base point is handed in (the results are
            # up to date), free[0] "-" and every other job point
            # equal the base there, so they run clean.
            base_value: float = win._param_panel.param_values()[free[0]]

            def flaky(
                x: np.ndarray, _input_data: gui_data.GuiInputData
            ) -> gui_data.GuiRunResults:
                if float(x[failing_index]) > base_value:
                    raise ValueError("point failed")
                return plain_fake(x, _input_data)

            monkeypatch.setattr(
                gui_data,
                "arr_to_results",
                _file_gated_fake(flaky, gate),
            )

            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window._GradientWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            _wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            _pump()
            # The job's points ran in the pool (their `calls`
            # appends never reach this process): only the
            # in-process startup run is recorded, and the last
            # step completes the bar.
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) <= total <= 2 * len(free)

            assert win._gradient_state is not None
            state: gui_window._GradientSnapshot = win._gradient_state
            table: pd.DataFrame = state.table
            # The failed point's column is all NaN (the missing
            # value); every other column is fully finite.
            assert np.isnan(table.values).any()
            assert np.isnan(table[free[0]].values).all()
            for key in free[1:]:
                assert np.isfinite(table[key].values).all()

            # The completion status keeps its exact wording and
            # appends the failed-point count.
            assert (
                win._status_label.text()
                == f"Gradients finished in {state.elapsed_seconds:.1f} s — "
                f"saved to {state.csv_path}."
                f" 1 of {total} points failed; their values are shown "
                f"as missing."
            )
            assert win._figure_panel.page_count() == 9
            assert win._figure_panel._titles[-1] == "Gradients (full)"
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            assert win._gradients_button.isEnabled()
            _only_csv(tmp_path)
        finally:
            gate.touch()
            _close_window(win)

    def test_gradient_job_fails_when_base_point_fails(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A failed base point still fails the whole job: the
        matrix cannot be built without it, so the job reports
        the failure and writes no matrix.
        """
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        plain_fake = _fake_results_factory(calls, input_data)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", plain_fake)
        win = gui_window.MainWindow(input_data)
        gate = tmp_path / "job_gate"
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            # Move free[0] to its upper bound, with the auto run
            # off so the edit does not start a model run: the
            # results go stale, so the job must run the base point
            # itself.  The base's "+" perturbation clamps at the
            # bound and coincides with the base, so it is
            # deduplicated out of the pool points and only the
            # base fails.
            win._auto_run_checkbox.setChecked(False)
            row = win._param_panel._rows[free[0]]
            assert isinstance(row, gui_param_panel._FreeCell)
            row._spin.setValue(win._param_panel.current_bounds()[free[0]][1])
            assert not win._results_are_up_to_date()
            base_x: np.ndarray = win._current_x()

            def flaky(
                x: np.ndarray, _input_data: gui_data.GuiInputData
            ) -> gui_data.GuiRunResults:
                if np.array_equal(x, base_x):
                    raise ValueError("point failed")
                return plain_fake(x, _input_data)

            monkeypatch.setattr(
                gui_data,
                "arr_to_results",
                _file_gated_fake(flaky, gate),
            )

            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window._GradientWorker)
            gate.touch()

            _wait_until(
                lambda: win._worker is None
                and win._status_label.text().startswith("Gradients failed")
            )
            _pump()
            assert (
                win._status_label.text()
                == "Gradients failed: ValueError: point failed"
            )
            assert win._gradient_state is None
            assert win._figure_panel.page_count() == 8
            # The base never arrived, so the results stay stale:
            # the save button stays down and the run button stays
            # up.
            assert not win._save_button.isEnabled()
            assert win._run_button.isEnabled()
            assert win._gradients_button.isEnabled()
            # The job's points ran in the pool (their `calls`
            # appends never reach this process): only the
            # in-process startup run is recorded, and no matrix
            # was written.
            assert len(calls) == 1
            assert list((tmp_path / "results").glob("gradients_*.csv")) == []
        finally:
            gate.touch()
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

    def test_gradient_job_progress_step_counts(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A gradient job reports each completed point as a
        (done, total) step count, so the bar is determinate and the
        header's spinner spins while the job runs; both go back to
        their rest state when the job finishes.
        """
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        win = _gradient_window(
            input_data, tmp_path, monkeypatch, full_fake, hydro_fake
        )
        gate = tmp_path / "job_gate"
        try:
            # Let the window's startup run finish first, so the
            # job's base point is handed in rather than rerun.
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert win._results_are_up_to_date()

            # The job's pool points block on a file gate (visible
            # to the forked workers) until the test has connected
            # to the worker's progress_step signal: the worker is
            # created by the click, so it cannot be connected in
            # advance, and the gate keeps it from completing (and
            # emitting a step for) a point before the connection.
            monkeypatch.setattr(
                gui_data,
                "arr_to_results",
                _file_gated_fake(full_fake, gate),
            )
            monkeypatch.setattr(
                gui_data,
                "arr_to_hydro_results",
                _file_gated_fake(hydro_fake, gate),
            )

            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window._GradientWorker)

            steps: list[tuple[int, int]] = []
            spinner_up: list[bool] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))
                spinner_up.append(not win._job_spinner.isHidden())

            worker.progress_step.connect(on_step)
            gate.touch()

            _wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            _pump()

            # Every completed point is one step, in order, and the
            # last step completes the bar.
            assert len(steps) > 0
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)

            # The base was handed in, so the total is the deduped
            # number of perturbed points: at most one point per
            # parameter coincides with the base, and each
            # parameter contributes at least one of its two.
            assert len(free) <= total <= 2 * len(free)

            # The spinner is up for every step and down after the
            # job, and the bar is back to its indeterminate rest
            # state with the strip hidden.
            assert all(spinner_up)
            assert win._job_spinner.isHidden()
            assert win._progress_bar.maximum() == 0
            assert not win._progress_bar.isTextVisible()
            assert win._progress_strip.isHidden()
        finally:
            gate.touch()
            _close_window(win)


# ------------------------------------------------------------------ #
# The objective table's checkbox selection: the search objective
# ------------------------------------------------------------------ #


class TestObjectiveSelection:
    """Checking objective-table cells picks the line search objective."""

    def test_objective_selection_toggles(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Clicking a selectable cell toggles its term in the
        window's selection and in the model; a non-selectable
        term is ignored.
        """
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

            viewport = win._objective_table.viewport()
            index = win._objective_model.index(0, 0)
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(win._objective_table.visualRect(index).center()),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            qapp.sendEvent(viewport, event)
            _pump()
            assert win._objective_terms == frozenset({"Streamflow: KGE"})
            assert win._objective_model.is_selected("Streamflow: KGE")

            qapp.sendEvent(viewport, event)
            _pump()
            assert win._objective_terms == frozenset()
            assert not win._objective_model.is_selected("Streamflow: KGE")

            win._objective_terms = None
            win._on_objective_toggled("DOC: R\u00b2")
            assert win._objective_terms is None
        finally:
            _close_window(win)

    def test_mode_switch_keeps_explicit_selection_intersected(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Switching the run mode intersects an explicit selection
        with the mode's selectable set and repaints the
        checkboxes; switching back keeps the surviving term
        selected while the dropped term is present but unchecked.
        """
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
            win._auto_run_checkbox.setChecked(False)

            win._on_objective_toggled("Streamflow: KGE")
            win._on_objective_toggled("DOC: KGE")
            assert win._objective_terms == frozenset(
                {"Streamflow: KGE", "DOC: KGE"}
            )

            win._mode_checkbox.setChecked(True)
            assert win._objective_terms == frozenset({"Streamflow: KGE"})
            assert win._objective_model.is_selected("Streamflow: KGE")
            assert not win._objective_model.is_selected("DOC: KGE")

            win._mode_checkbox.setChecked(False)
            assert win._objective_terms == frozenset({"Streamflow: KGE"})
            assert not win._objective_model.is_selected("DOC: KGE")
        finally:
            _close_window(win)


# ------------------------------------------------------------------ #
# The line search's pure helpers: the feasible-direction projection
# and the snapped search grid
# ------------------------------------------------------------------ #


class TestLineSearchHelpers:
    """The pure helpers the line search's worker relies on."""

    def test_project_direction_zeroes_outward_bound_components(
        self,
    ) -> None:
        """
        A component pointing out of a bound the base point sits on
        is dropped, and the surviving vector is renormalized to
        unit length: inward and unblocked components keep their
        direction.
        """
        from ..gui.line_search import project_direction

        bounds: dict[str, tuple[float, float]] = {
            "a": (0.0, 1.0),
            "b": (0.0, 2.0),
            "c": (0.5, 1.5),
            "d": (0.0, 3.0),
            "e": (0.0, 1.0),
        }
        # `a` sits on its lower bound and points down (blocked),
        # `c` sits on its upper bound and points up (blocked),
        # `e` sits on its lower bound but points in (kept).
        base_x: np.ndarray = np.array([0.0, 1.0, 1.5, 2.0, 0.0])
        raw: np.ndarray = np.array([-1.0, 2.0, 3.0, 4.0, 1.0])
        expected: np.ndarray = np.array([0.0, 2.0, 0.0, 4.0, 1.0])
        expected = expected / float(np.linalg.norm(expected))
        projected: np.ndarray = project_direction(
            base_x,
            raw,
            free_parameter_steps(base_x, bounds),
            bounds,
        )
        np.testing.assert_allclose(projected, expected)
        assert float(np.linalg.norm(projected)) == pytest.approx(1.0)

    def test_project_direction_fails_when_no_component_survives(
        self,
    ) -> None:
        """
        When every free parameter sits on the bound its component
        points out of, the projection is the zero vector: the
        search has no direction to follow, and the failure names
        the blocked parameters.
        """
        from ..gui.line_search import project_direction

        bounds: dict[str, tuple[float, float]] = {
            "a": (0.0, 1.0),
            "b": (0.5, 1.5),
        }
        base_x: np.ndarray = np.array([0.0, 1.5])
        raw: np.ndarray = np.array([-1.0, 1.0])
        with pytest.raises(ValueError) as excinfo:
            project_direction(
                base_x,
                raw,
                free_parameter_steps(base_x, bounds),
                bounds,
            )
        message: str = str(excinfo.value)
        assert "no feasible direction" in message
        assert "a" in message
        assert "b" in message

    def test_search_t_points_snaps_the_nearest_point_to_zero(
        self,
    ) -> None:
        """
        The grid is equally spaced across the feasible interval with
        the nearest point snapped to ``t = 0``: a grid that would
        miss the base point exactly still evaluates it, and the
        snapped grid stays strictly increasing.
        """
        from ..gui.line_search import search_t_points

        symmetric: np.ndarray = search_t_points(-1.0, 1.0, 5)
        np.testing.assert_allclose(
            symmetric, np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        )
        snapped: np.ndarray = search_t_points(-1.0, 2.0, 5)
        np.testing.assert_allclose(
            snapped, np.array([-1.0, 0.0, 0.5, 1.25, 2.0])
        )
        assert (np.diff(snapped) > 0).all()


# ------------------------------------------------------------------ #
# The header's *Line search* button: the steepest-ascent jobs
# ------------------------------------------------------------------ #


class TestLineSearch:
    """The header's *Line search* button: the steepest-ascent jobs."""

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
        gate = tmp_path / "job_gate"
        win = _gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
            gate_path=gate,
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

            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            worker = win._worker
            assert isinstance(worker, gui_window._LineSearchWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            _wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            _pump()
            # The base point's results were handed in, the
            # direction came from its own central differences
            # (the first pool phase), then the 25 search points
            # were run (the second): all the job's points ran in
            # the pool, so only the in-process startup run is
            # recorded, and the last step completes the bar.
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) + 25 <= total <= 2 * len(free) + 25

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
            assert win._figure_panel.page_count() == 9
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

    def test_line_search_job_failed_point_is_nan_in_profile(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A failed search point does not fail the job: its profile
        row's value is NaN (the missing value), the best point is
        chosen among the finite values, and the completion status
        counts the failure.
        """
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = tmp_path / "job_gate"
        win = _gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
            gate_path=gate,
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            # The fake shifts the streamflow objectives by 2 * x of
            # free[0] alone, so the steepest ascent of Streamflow's
            # KGE is the unit direction along free[0]'s axis and
            # the best point is free[0]'s upper bound.
            mid: float = win._param_panel.param_values()[free[0]]
            high: float = win._param_panel.current_bounds()[free[0]][1]

            # The window has no gradient job yet, so the worker
            # computes its own direction: replay the same central
            # differences with a separate fake (the precomputed
            # points' calls must not land in `calls`) to
            # precompute the 25 search points bit for bit.
            pre_calls: list[np.ndarray] = []
            pre_full, _pre_hydro = _gradient_fake_factory(
                pre_calls, input_data, free[0], free[1]
            )
            base_x: np.ndarray = win._current_x()
            bounds: dict[str, tuple[float, float]] = (
                win._param_panel.current_bounds()
            )
            search_steps: list[_Perturbation] = free_parameter_steps(
                base_x, bounds
            )
            f_plus: list[np.ndarray] = [
                objective_vector(pre_full(step.x_plus, input_data))[1]
                for step in search_steps
            ]
            f_minus: list[np.ndarray] = [
                objective_vector(pre_full(step.x_minus, input_data))[1]
                for step in search_steps
            ]
            assert win._results is not None
            base_labels, _ = objective_vector(win._results)
            row_index: int = base_labels.index("Streamflow: KGE")
            matrix: np.ndarray = gradient_matrix(
                f_plus, f_minus, search_steps
            )
            direction: np.ndarray = expand_direction(
                matrix[row_index]
                / float(np.linalg.norm(matrix[row_index])),
                search_steps,
                bounds,
            )
            interval = feasible_t_interval(
                base_x, direction, search_steps, bounds
            )
            assert interval is not None
            t_lo, t_hi = interval
            from ..gui.line_search import search_t_points

            # The same grid the worker builds: a linspace over the
            # feasible interval with the point nearest the base
            # snapped onto it (a no-op for this symmetric interval).
            t_points: np.ndarray = search_t_points(
                t_lo, t_hi, gui_window._LINE_SEARCH_POINTS
            )
            search_points: list[np.ndarray] = [
                base_x + t * direction for t in t_points
            ]
            # The point that fails is a non-base grid point: the
            # base point (t = 0) is no longer run in the pool, so it
            # cannot be the one that fails.
            failed_index: int = 5

            # The search's own pool phase is gated on its own file,
            # re-patched for the search's workers: the fake raises
            # for the one flagged search point and delegates to the
            # startup fake otherwise.
            def flaky_full(
                x: np.ndarray, _input_data: gui_data.GuiInputData
            ) -> gui_data.GuiRunResults:
                if np.array_equal(x, search_points[failed_index]):
                    raise ValueError("point failed")
                return full_fake(x, _input_data)

            def flaky_hydro(
                x: np.ndarray, _input_data: gui_data.GuiInputData
            ) -> gui_data.GuiRunResults:
                if np.array_equal(x, search_points[failed_index]):
                    raise ValueError("point failed")
                return hydro_fake(x, _input_data)

            monkeypatch.setattr(
                gui_data,
                "arr_to_results",
                _file_gated_fake(flaky_full, gate),
            )
            monkeypatch.setattr(
                gui_data,
                "arr_to_hydro_results",
                _file_gated_fake(flaky_hydro, gate),
            )

            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            worker = win._worker
            assert isinstance(worker, gui_window._LineSearchWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            _wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            _pump()
            # The base point's results were handed in, the
            # direction came from its own central differences
            # (the first pool phase), then the 25 search points
            # were run (the second): all the job's points ran in
            # the pool, so only the in-process startup run is
            # recorded, and the last step completes the bar.
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) + 25 <= total <= 2 * len(free) + 25

            assert win._line_search_state is not None
            state: gui_window._LineSearchSnapshot = win._line_search_state
            profile: pd.DataFrame = state.profile
            assert len(profile) == 25
            values: np.ndarray = profile["value"].to_numpy(dtype=np.float64)
            # The failed search point is NaN (the missing value),
            # and every other point reports a finite value.
            assert np.isnan(values[failed_index])
            assert np.isfinite(values).sum() == len(profile) - 1

            # The best point is chosen among the finite values:
            # the objective is monotone in t, so it is the last
            # point, free[0]'s upper bound, adopted into the
            # panel.
            best_index: int = int(np.nanargmax(values))
            assert best_index == len(profile) - 1
            assert float(profile.index[best_index]) == pytest.approx(
                high - mid
            )
            assert (
                win._param_panel.param_values()[free[0]]
                == pytest.approx(high)
            )
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()

            # The completion status keeps its exact wording and
            # appends the failed-point count.
            best_t: float = float(profile.index[best_index])
            best_value: float = float(values[best_index])
            assert (
                win._status_label.text()
                == f"Line search finished in {state.elapsed_seconds:.1f} s: "
                f"best t = {best_t:.4f} (Streamflow: KGE = {best_value:.3g}); "
                f"the best point's parameters are in the panel and the "
                f"profile was saved to {state.csv_path}."
                f" 1 of {total} points failed; their values are shown "
                f"as missing."
            )

            # The profile is written once and rides along as the
            # figure panel's last page.
            csvs: list[Path] = list(
                (tmp_path / "results").glob("line_search_*.csv")
            )
            assert len(csvs) == 1
            assert state.csv_path is not None
            assert tmp_path / state.csv_path == csvs[0]
            assert win._figure_panel.page_count() == 9
            assert win._figure_panel._titles[-1] == "Line search (full)"
        finally:
            gate.touch()
            _close_window(win)

    def test_worker_reuses_the_base_results_at_the_zero_step(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        The grid point at ``t = 0`` is the base point itself: the
        worker reuses the handed-in base results for that profile
        row instead of running the point in the pool, and every
        other grid point is pooled exactly once.
        """
        monkeypatch.chdir(tmp_path)
        bounds: dict[str, tuple[float, float]] = dict(input_data.bounds)
        base_x: np.ndarray = _midpoint_vector(input_data)
        steps: list[_Perturbation] = free_parameter_steps(base_x, bounds)
        free = _free_keys(input_data)
        keys: list[str] = list(bounds)
        direction: np.ndarray = np.zeros(len(keys))
        direction[keys.index(free[0])] = 0.6
        direction[keys.index(free[1])] = 0.8
        hydro, rt = _synthetic_results(input_data)
        base_results: gui_data.GuiRunResults = gui_data.GuiRunResults(
            hydro, rt
        )

        received: list[np.ndarray] = []

        def pool_fake(
            points: list[np.ndarray],
            _mode: str,
            _csv_path: str,
            workers: int | None = None,
            on_result: (
                Callable[[int, gui_data.GuiRunResults | str], None] | None
            ) = None,
            model_key: str = "lateral_three_layer",
            network: pt.ReactionNetwork | None = None,
        ) -> list[gui_data.GuiRunResults | str]:
            payloads: list[gui_data.GuiRunResults | str] = []
            for index, point in enumerate(points):
                received.append(np.array(point))
                point_hydro, point_rt = _synthetic_results(input_data)
                point_hydro.objective_functions = (
                    point_hydro.objective_functions + 0.1
                )
                payload: gui_data.GuiRunResults = gui_data.GuiRunResults(
                    point_hydro, point_rt
                )
                payloads.append(payload)
                if on_result is not None:
                    on_result(index, payload)
            return payloads

        monkeypatch.setattr(gui_window, "map_model_runs", pool_fake)
        worker: gui_window._LineSearchWorker = gui_window._LineSearchWorker(
            base_x,
            steps,
            direction,
            ["Streamflow: KGE"],
            MODE_FULL,
            str(gui_data.DEFAULT_CSV_PATH),
            bounds,
            base_results,
        )
        try:
            outcome = worker._compute(0.0)

            assert outcome.error is None
            assert outcome.profile is not None
            profile: pd.DataFrame = outcome.profile
            assert len(profile) == 25
            t_values: np.ndarray = profile.index.to_numpy(
                dtype=np.float64
            )
            zero_row: int = int(np.argmin(np.abs(t_values)))
            # The zero step sits exactly on the grid.
            assert t_values[zero_row] == 0.0
            values: np.ndarray = profile["value"].to_numpy(
                dtype=np.float64
            )
            # The zero row carries the base results' value (the
            # synthetic KGE), every pooled point the shifted one.
            assert values[zero_row] == pytest.approx(0.9)
            for row in range(len(t_values)):
                if row != zero_row:
                    assert values[row] == pytest.approx(1.0)
            # The pool saw exactly the non-base points, once each.
            assert len(received) == 24
            for point in received:
                assert not np.array_equal(point, base_x)
        finally:
            worker.deleteLater()
            _pump()

    def test_no_improvement_keeps_the_base_results(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A profile whose best value equals the base value (the search
        found no improvement) is kept as the figure panel's page
        without adopting anything: the panel, the results, and the
        last-run vector stay where they are, and the completion
        status says the parameters are unchanged.
        """
        from ..gui.line_search import write_line_search_csv

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

            before_values: dict[str, float] = (
                win._param_panel.param_values()
            )
            assert win._results is not None
            before_results: gui_data.GuiRunResults = win._results
            before_x: np.ndarray | None = win._last_run_x
            base_x: np.ndarray = win._current_x()

            # A profile that peaks at the base point: its best
            # value is the base value, so the search found no
            # improvement.
            t_points: np.ndarray = np.linspace(-0.5, 0.5, 25)
            t_points[int(np.argmin(np.abs(t_points)))] = 0.0
            values: np.ndarray = 0.9 - 0.01 * np.abs(t_points)
            profile: pd.DataFrame = pd.DataFrame(
                {"value": values, "Streamflow: KGE": values},
                index=t_points,
            )
            csv_path: Path = write_line_search_csv(profile)
            alt_hydro, alt_rt = _synthetic_results(input_data)
            outcome = gui_window._LineSearchOutcome(
                error=None,
                profile=profile,
                x_best=base_x,
                best_results=gui_data.GuiRunResults(alt_hydro, alt_rt),
                csv_path=csv_path,
                terms=["Streamflow: KGE"],
                mode=win._current_mode(),
                model_key=model_specs.DEFAULT_MODEL_KEY,
                elapsed_seconds=3.0,
                failed_points=0,
                total_points=25,
            )
            win._on_finished_line_search(outcome)

            # The base point's results stay on screen, and the
            # profile is the figure panel's last page.
            assert win._param_panel.param_values() == before_values
            assert win._results is before_results
            assert win._last_run_x is before_x
            assert win._last_run_mode == MODE_FULL

            assert win._line_search_state is not None
            state: gui_window._LineSearchSnapshot = win._line_search_state
            assert state.profile is profile
            assert state.csv_path == csv_path
            assert win._figure_panel.page_count() == 9
            assert win._figure_panel._titles[-1] == "Line search (full)"

            assert win._line_search_button.isEnabled()
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            assert win._status_label.text() == (
                "Line search finished in 3.0 s: no improvement found "
                "(the best value equals the base value); the "
                "parameters are unchanged and the profile was saved "
                f"to {csv_path}."
            )
        finally:
            _close_window(win)

    def test_improvement_is_still_adopted(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A profile whose best value beats the base value keeps the
        existing adoption: the best point's parameters and results
        move into the window, and the completion status reports the
        best step and value.
        """
        from ..gui.line_search import write_line_search_csv

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

            before_values: dict[str, float] = (
                win._param_panel.param_values()
            )
            base_x: np.ndarray = win._current_x()
            free = _free_keys(input_data)
            keys: list[str] = list(win._param_panel.current_bounds())
            x_best: np.ndarray = np.array(base_x)
            x_best[keys.index(free[0])] += 0.001

            # A profile that keeps rising with t: its best point
            # beats the base point's value.
            t_points: np.ndarray = np.linspace(-0.5, 0.5, 25)
            t_points[int(np.argmin(np.abs(t_points)))] = 0.0
            values: np.ndarray = 0.9 + 0.01 * t_points
            profile: pd.DataFrame = pd.DataFrame(
                {"value": values, "Streamflow: KGE": values},
                index=t_points,
            )
            csv_path: Path = write_line_search_csv(profile)
            alt_hydro, alt_rt = _synthetic_results(input_data)
            alt_results: gui_data.GuiRunResults = gui_data.GuiRunResults(
                alt_hydro, alt_rt
            )
            outcome = gui_window._LineSearchOutcome(
                error=None,
                profile=profile,
                x_best=x_best,
                best_results=alt_results,
                csv_path=csv_path,
                terms=["Streamflow: KGE"],
                mode=win._current_mode(),
                model_key=model_specs.DEFAULT_MODEL_KEY,
                elapsed_seconds=3.0,
                failed_points=0,
                total_points=25,
            )
            win._on_finished_line_search(outcome)

            after_values: dict[str, float] = (
                win._param_panel.param_values()
            )
            for key in before_values:
                expected: float = (
                    before_values[key] + 0.001
                    if key == free[0]
                    else before_values[key]
                )
                assert after_values[key] == pytest.approx(expected)

            assert win._results is alt_results
            last_x: np.ndarray | None = win._last_run_x
            assert last_x is not None
            assert np.array_equal(last_x, x_best)
            assert win._last_run_mode == MODE_FULL

            assert win._line_search_state is not None
            state: gui_window._LineSearchSnapshot = win._line_search_state
            assert state.profile is profile
            assert state.csv_path == csv_path
            assert win._figure_panel.page_count() == 9
            assert win._figure_panel._titles[-1] == "Line search (full)"

            assert win._line_search_button.isEnabled()
            assert win._save_button.isEnabled()
            assert not win._run_button.isEnabled()
            assert win._status_label.text() == (
                f"Line search finished in 3.0 s: best t = 0.5000 "
                f"(Streamflow: KGE = 0.905); the best point's "
                f"parameters are in the panel and the profile was "
                f"saved to {csv_path}."
            )
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
        gradient_gate = tmp_path / "gradient_gate"
        win = _gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
            gate_path=gradient_gate,
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window._GradientWorker)

            gradient_steps: list[tuple[int, int]] = []

            def on_gradient_step(done: int, total: int) -> None:
                gradient_steps.append((done, total))

            worker.progress_step.connect(on_gradient_step)
            gradient_gate.touch()
            _wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            _pump()
            # The job's points ran in the pool (their `calls`
            # appends never reach this process): only the
            # in-process startup run is recorded, and the last
            # step completes the bar.
            calls_after_gradient: int = len(calls)
            assert calls_after_gradient == 1
            assert gradient_steps
            assert all(
                a[0] < b[0]
                for a, b in zip(gradient_steps, gradient_steps[1:])
            )
            assert len({total for _done, total in gradient_steps}) == 1
            total: int = gradient_steps[0][1]
            assert gradient_steps[-1] == (total, total)
            assert len(free) <= total <= 2 * len(free)

            mid: float = win._param_panel.param_values()[free[0]]
            high: float = win._param_panel.current_bounds()[free[0]][1]

            # The search's own pool phase is gated on its own file,
            # re-patched for the search's workers.
            search_gate = tmp_path / "search_gate"
            monkeypatch.setattr(
                gui_data,
                "arr_to_results",
                _file_gated_fake(full_fake, search_gate),
            )
            monkeypatch.setattr(
                gui_data,
                "arr_to_hydro_results",
                _file_gated_fake(hydro_fake, search_gate),
            )

            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            search_worker = win._worker
            assert isinstance(search_worker, gui_window._LineSearchWorker)

            search_steps: list[tuple[int, int]] = []

            def on_search_step(done: int, total: int) -> None:
                search_steps.append((done, total))

            search_worker.progress_step.connect(on_search_step)
            search_gate.touch()
            _wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            _pump()
            # No direction differences and no base re-run: the
            # snapshot's row supplies the direction and the base
            # point's results are handed in, so the only pool
            # points are the 25 search points — and none of them
            # run in-process, so the recorded calls stay put.
            assert len(calls) == 1
            assert search_steps
            assert all(
                a[0] < b[0] for a, b in zip(search_steps, search_steps[1:])
            )
            assert len({total for _done, total in search_steps}) == 1
            assert search_steps[-1] == (25, 25)

            assert win._line_search_state is not None
            profile: pd.DataFrame = win._line_search_state.profile
            assert len(profile) == 25
            # The snapshot's row is the raw difference, 2.0 along
            # free[0]'s axis; the search renormalizes it to a unit
            # direction, so the farthest step spans the whole
            # remaining distance to the bound — the same upper
            # bound the point lands on either way.
            assert float(profile.index.max()) == pytest.approx(high - mid)
            assert (
                win._param_panel.param_values()[free[0]]
                == pytest.approx(high)
            )

            # Both jobs' pages ride along, in order.
            assert win._figure_panel.page_count() == 10
            assert win._figure_panel._titles[-2:] == [
                "Gradients (full)",
                "Line search (full)",
            ]
        finally:
            _close_window(win)

    def test_line_search_sum_of_two_objectives(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Two checked cells search the sum of the two objectives:
        the profile carries a column per term and their sum in
        the value column, and the page and the status report on
        the sum.
        """
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = tmp_path / "search_gate"
        win = _gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
            gate_path=gate,
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            terms: frozenset[str] = frozenset(
                {"Streamflow: KGE", "Streamflow: NSE"}
            )
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            worker = win._worker
            assert isinstance(worker, gui_window._LineSearchWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            _wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            _pump()
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) + 25 <= total <= 2 * len(free) + 25

            state = win._line_search_state
            assert state is not None
            assert state.mode == MODE_FULL
            assert state.terms == ["Streamflow: KGE", "Streamflow: NSE"]
            profile: pd.DataFrame = state.profile
            assert len(profile) == 25
            assert profile.columns.tolist()[:3] == [
                "value",
                "Streamflow: KGE",
                "Streamflow: NSE",
            ]
            assert np.array_equal(
                profile["value"],
                profile["Streamflow: KGE"] + profile["Streamflow: NSE"],
            )

            assert win._figure_panel.page_count() == 9
            assert win._figure_panel._titles[-1] == "Line search (full)"
            figure: Figure = win._figure_panel._canvases[-1].figure
            lines = _lines(figure.axes[0])
            assert [line.get_label() for line in lines] == [
                "Streamflow: KGE",
                "Streamflow: NSE",
                "sum",
                "best",
            ]
            assert lines[2].get_linewidth() == 2
            assert [t.get_text() for t in figure.legends[0].get_texts()] == [
                "Streamflow: KGE",
                "Streamflow: NSE",
                "sum",
                "best",
            ]
            assert re.fullmatch(
                r"Line search finished in \d+(?:\.\d+)? s: "
                r"best t = [\d.]+ \(sum of 2 objectives = [\d.]+\); "
                r"the best point's parameters are in the panel and the "
                r"profile was saved to .*line_search_\d{8}_\d{6}\.csv\.",
                win._status_label.text(),
            ) is not None
        finally:
            gate.touch()
            _close_window(win)

    def test_line_search_in_hydro_mode(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        The same checkbox selection searches in hydro mode: the
        pool runs the hydro objectives and the page and the
        status report the hydro search.
        """
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        win = _make_window(input_data, tmp_path, monkeypatch, full_fake)
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            # No auto run after the mode switch: the search's
            # points are the only pool work, held on the file
            # gate until the test releases them.
            win._auto_run_checkbox.setChecked(False)
            win._mode_checkbox.setChecked(True)
            file_gate = tmp_path / "job_gate"

            def marking_hydro_fake(
                x: np.ndarray, _input_data: gui_data.GuiInputData
            ) -> gui_data.GuiRunResults:
                (tmp_path / f"hydro_{float(x.sum()):.6f}").touch()
                return hydro_fake(x, _input_data)

            monkeypatch.setattr(
                gui_data,
                "arr_to_hydro_results",
                _file_gated_fake(marking_hydro_fake, file_gate),
            )
            monkeypatch.setattr(
                gui_data,
                "arr_to_results",
                _file_gated_fake(full_fake, file_gate),
            )

            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            assert isinstance(
                win._worker, gui_window._LineSearchWorker
            )

            file_gate.touch()
            _wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            _pump()
            markers: list[Path] = sorted(tmp_path.glob("hydro_*"))
            assert markers

            state = win._line_search_state
            assert state is not None
            assert state.mode == MODE_HYDRO
            assert state.terms == ["Streamflow: KGE"]
            profile: pd.DataFrame = state.profile
            assert np.allclose(profile["value"], 0.9 + 2.0 * profile[free[0]])

            # Hydro mode's own pages are the streamflow figure and
            # the water-storage figure: two, plus the search page.
            assert win._figure_panel.page_count() == 4
            assert win._figure_panel._titles[-1] == "Line search (hydro)"
            assert re.fullmatch(
                r"Line search finished in \d+(?:\.\d+)? s: "
                r"best t = [\d.]+ \(Streamflow: KGE = [\d.]+\); "
                r"the best point's parameters are in the panel and the "
                r"profile was saved to .*line_search_\d{8}_\d{6}\.csv\.",
                win._status_label.text(),
            ) is not None
        finally:
            file_gate.touch()
            _close_window(win)

    def test_line_search_recomputes_direction_when_snapshot_lacks_term(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        When the saved gradient snapshot lacks a selected term, the
        search recomputes the direction from the base point's own
        differences instead of reusing the snapshot's rows.
        """
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = tmp_path / "search_gate"
        win = _gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
            gate_path=gate,
        )
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert len(calls) == 1

            # A snapshot that lacks the selected term, with a csv
            # file that exists so only the term lookup can fail.
            results_dir: Path = tmp_path / "results"
            results_dir.mkdir(exist_ok=True)
            csv_path: Path = results_dir / "gradients_fake.csv"
            csv_path.touch()
            win._gradient_state = gui_window._GradientSnapshot(
                table=pd.DataFrame(
                    [[0.0] * len(free)],
                    index=["Streamflow: KGE"],
                    columns=free,
                ),
                base_x=win._current_x(),
                mode=MODE_FULL,
                csv_path=csv_path,
                elapsed_seconds=0.0,
                model_key=model_specs.DEFAULT_MODEL_KEY,
            )

            terms: frozenset[str] = frozenset({"DOC: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            worker = win._worker
            assert isinstance(worker, gui_window._LineSearchWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            _wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            _pump()
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            # The base point's results were handed in and the
            # midpoint base dedups against nothing, so the
            # direction's central differences run for every free
            # parameter, then the 25 search points follow.
            assert total == 2 * len(free) + 25

            state = win._line_search_state
            assert state is not None
            assert state.mode == MODE_FULL
            assert state.terms == ["DOC: KGE"]
            assert re.fullmatch(
                r"Line search finished in \d+(?:\.\d+)? s: "
                r"best t = [\d.]+ \(DOC: KGE = [\d.]+\); "
                r"the best point's parameters are in the panel and the "
                r"profile was saved to .*line_search_\d{8}_\d{6}\.csv\.",
                win._status_label.text(),
            ) is not None
        finally:
            gate.touch()
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
            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
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

    def test_line_search_refused_while_a_job_is_in_flight(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        The direct path: a line search invoked while a job
        is in flight is refused with a status message, and the
        in-flight job's outcome stays the final on-screen state;
        with no job in flight, the same call starts the search.
        """
        free = _free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, _hydro_fake = _gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = threading.Event()
        file_gate = tmp_path / "job_gate"

        def gated_fake(
            x: np.ndarray, _input_data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            # The gate blocks the model calls until the test
            # releases it, so the run stays in flight while the
            # refusal is asserted, and then finishes on demand.
            gate.wait(timeout=60)
            return full_fake(x, _input_data)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", gated_fake)
        monkeypatch.setattr(gui_window, "arr_to_hydro_results", gated_fake)
        # The search job's points run in a forked pool, whose
        # workers resolve the fakes from the `data` module. The
        # file gate holds them until the test releases it.
        monkeypatch.setattr(
            gui_data,
            "arr_to_results",
            _file_gated_fake(full_fake, file_gate),
        )
        monkeypatch.setattr(
            gui_data,
            "arr_to_hydro_results",
            _file_gated_fake(full_fake, file_gate),
        )
        win = gui_window.MainWindow(input_data)
        try:
            # The window's startup run blocks on the gate: a model
            # run in flight, the only job the window knows.
            _wait_until(
                lambda: win._worker is not None
                and win._worker.isRunning()
            )
            assert isinstance(win._worker, gui_window._RunWorker)
            run_id: int = id(win._worker)

            # The direct path is guarded against the in-flight run:
            # refused with a status message, the run's worker
            # reference untouched, nothing adopted or started.
            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            _pump()
            assert win._worker is not None
            assert id(win._worker) == run_id
            assert win._worker.isRunning()
            assert not isinstance(win._worker, gui_window._LineSearchWorker)
            assert win._line_search_state is None
            assert win._results is None
            assert win._status_label.text() == (
                "A job is already in flight: it cannot be interrupted, "
                "so the line search was not started."
            )
            assert len(calls) == 0

            # Releasing the gate lets the run finish; its outcome is
            # the final on-screen state (up to date, and no line
            # search was ever started to clobber it).
            gate.set()
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            assert win._results_are_up_to_date()
            assert win._line_search_state is None
            assert win._figure_panel.page_count() == 8
            assert re.fullmatch(
                r"Run 1 finished in \d+\.\d s\.", win._status_label.text()
            ) is not None
            assert len(calls) == 1
            assert np.array_equal(calls[0], _midpoint_vector(input_data))

            # With no job in flight, the same call starts the
            # search: the base point's results are handed in, and
            # the worker computes the direction and the search
            # points itself in the pool.
            file_gate.touch()
            terms = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            assert isinstance(win._worker, gui_window._LineSearchWorker)
            _wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            _pump()
            # Only the in-process startup run is recorded: the
            # job's points ran in forked workers, whose calls
            # never reach this process's list.
            assert len(calls) == 1
        finally:
            gate.set()
            file_gate.touch()
            _close_window(win)


class TestCalibration:
    """The header's *Calibrate* button and its differential-evolution job."""

    def test_metric_keys_for_terms(self) -> None:
        """Only the supported metrics map, in first-seen order."""
        assert gui_calibration.metric_keys_for_terms(
            [
                "Streamflow: KGE",
                "Streamflow: LogKGE",
                "Streamflow: KGE",
                "Streamflow: Bias",
                "bogus",
            ]
        ) == ("kge", "log_kge")

    def test_objective_score(
        self, input_data: gui_data.GuiInputData
    ) -> None:
        """The score negates the equal-weighted metric sum."""
        hydro, _rt = _synthetic_results(input_data)
        expected: float = -float(
            hydro.objective_functions["kge"]
            + hydro.objective_functions["nse"]
        )
        assert gui_calibration.objective_score(
            hydro, ("kge", "nse")
        ) == pytest.approx(expected)
        assert (
            gui_calibration.objective_score(hydro, ("bogus",))
            == float("inf")
        )

    def test_worker_passes_options_and_expands_pinned(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        The worker searches the free hydrology parameters only, hands
        the pinned values in as the fixed vector, forwards the
        options, and reports the best full vector.
        """
        model_class = input_data.spec.model_class
        num_hydro: int = model_class.get_num_hydro_parameters()
        all_names: tuple[str, ...] = tuple(input_data.bounds)[:num_hydro]
        x: np.ndarray = _midpoint_vector(input_data)
        free_name: str = all_names[0]
        free_high: float = float(x[0]) + 0.1
        bounds: dict[str, tuple[float, float]] = {
            name: (float(x[index]), float(x[index]))
            for index, name in enumerate(all_names)
        }
        bounds[free_name] = (float(x[0]) - 0.1, free_high)

        hydro, _rt = _synthetic_results(input_data)
        records: list[dict[str, object]] = []

        def fake_calibration(cls: type, /, **kwargs: object) -> tuple[object, object, object]:
            records.append(kwargs)
            callback = kwargs["callback"]
            assert callable(callback)
            for generation in range(1, 3):
                callback(np.zeros(1), 0.5 / generation)
            metric = kwargs["metric"]
            assert callable(metric)
            assert metric(hydro) == gui_calibration.objective_score(
                hydro, ("kge", "nse")
            )
            return (
                {free_name: free_high},
                hydro,
                SimpleNamespace(fun=-2.5, nfev=123),
            )

        monkeypatch.setattr(
            model_class,
            "simple_calibration",
            classmethod(fake_calibration),
        )
        worker: gui_calibration._CalibrationWorker = (
            gui_calibration._CalibrationWorker(
                x,
                input_data,
                bounds,
                ["Streamflow: KGE", "Streamflow: NSE"],
                gui_calibration.CalibrationSettings(
                    generations=7, population_size=19, polish=True
                ),
                input_data.spec.key,
            )
        )
        steps: list[tuple[int, int]] = []
        lines: list[str] = []
        worker.progress_step.connect(
            lambda done, total: steps.append((done, total))
        )
        worker.progress_line.connect(lines.append)

        outcome = worker._compute()

        assert outcome.error is None
        assert outcome.x_best is not None
        assert outcome.x_best[0] == pytest.approx(free_high)
        # The pinned parameters keep their current values.
        for index in range(1, num_hydro):
            assert outcome.x_best[index] == pytest.approx(float(x[index]))
        assert outcome.results is not None
        assert outcome.results.rt is None
        assert outcome.best_value == pytest.approx(2.5)
        assert outcome.evaluations == 123
        assert outcome.terms == ["Streamflow: KGE", "Streamflow: NSE"]
        assert outcome.model_key == input_data.spec.key

        assert len(records) == 1
        kwargs = records[0]
        assert kwargs["maxiter"] == 7
        assert kwargs["popsize"] == 19
        assert kwargs["polish"] is True
        assert kwargs["num_threads"] == -1
        assert kwargs["param_ranges"] == {
            free_name: (float(x[0]) - 0.1, free_high)
        }
        base_params = kwargs["base_params"]
        assert isinstance(base_params, dict)
        for index, name in enumerate(all_names):
            assert base_params[name] == pytest.approx(float(x[index]))
        assert steps == [(0, 7), (1, 7), (2, 7)]
        assert lines[0].startswith("generation 1/7")

    def test_window_runs_adopts_and_saves(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        The button is available only in hydrology-only mode; a
        finished calibration adopts the best parameters, refreshes
        the results, and saves the run.
        """
        calls: list[np.ndarray] = []
        full_fake = _fake_results_factory(calls, input_data)
        hydro_fake = _fake_results_factory(calls, input_data, hydro_only=True)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", full_fake)
        monkeypatch.setattr(
            gui_window, "arr_to_hydro_results", hydro_fake
        )
        win = gui_window.MainWindow(input_data)
        try:
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )
            _pump()
            # Full mode: the button is disabled and the direct call
            # is refused with a message.
            assert not win._calibrate_button.isEnabled()
            win._on_calibrate_clicked()
            assert win._status_label.text() == (
                "Calibration optimizes the hydrology model only; "
                "switch the header's *Hydrology only* checkbox on."
            )
            assert win._worker is None

            # Hydrology-only mode: the button comes up after its run.
            win._mode_checkbox.setChecked(True)
            _wait_until(
                lambda: win._current_mode() == MODE_HYDRO
                and win._worker is None
                and win._results is not None
                and win._results.rt is None
            )
            _pump()
            assert win._calibrate_button.isEnabled()

            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            free_name: str = next(
                name
                for name in tuple(input_data.bounds)[
                    : input_data.spec.model_class.get_num_hydro_parameters()
                ]
                if win._param_panel.current_bounds()[name][1]
                > win._param_panel.current_bounds()[name][0]
            )
            high: float = win._param_panel.current_bounds()[free_name][1]
            hydro, _rt = _synthetic_results(input_data)

            def fake_calibration(cls: type, /, **kwargs: object) -> tuple[object, object, object]:
                callback = kwargs["callback"]
                assert callable(callback)
                callback(np.zeros(1), 0.1)
                return (
                    {free_name: high},
                    hydro,
                    SimpleNamespace(fun=-1.5, nfev=42),
                )

            monkeypatch.setattr(
                input_data.spec.model_class,
                "simple_calibration",
                classmethod(fake_calibration),
            )
            monkeypatch.setattr(
                gui_window,
                "ask_calibration_settings",
                lambda _parent: gui_calibration.CalibrationSettings(
                    generations=3, population_size=4, polish=False
                ),
            )
            win._on_calibrate_clicked()
            assert isinstance(win._worker, gui_calibration._CalibrationWorker)
            _wait_until(
                lambda: win._worker is None
                and "Calibration finished" in win._status_label.text()
            )
            _pump()

            assert win._param_panel.param_values()[free_name] == pytest.approx(
                high
            )
            assert win._last_run_mode == MODE_HYDRO
            assert win._results is not None
            assert win._results.rt is None
            assert win._results_are_up_to_date()
            assert "Calibration finished" in win._status_label.text()
            assert "saved to" in win._status_label.text()
            run_dir: Path = tmp_path / "results"
            saved: list[Path] = [
                path
                for path in run_dir.iterdir()
                if (path / "gui_state.json").is_file()
            ]
            assert len(saved) == 1
            assert saved[0].name in win._status_label.text()
            assert win._calibrate_button.isEnabled()
        finally:
            _close_window(win)


class TestModelSelection:
    """The header model selector and the state a switch resets."""

    def test_starts_on_lateral_and_titles_it(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The window opens on the lateral model and names it."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            assert win._model_combo.currentData() == "lateral_three_layer"
            assert win.windowTitle() == (
                "East River calibration — Lateral three-layer"
            )
        finally:
            _close_window(win)

    def test_model_combo_lists_the_four_models(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The selector offers exactly the four curated models, in order."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            labels: list[str] = [
                win._model_combo.itemText(i)
                for i in range(win._model_combo.count())
            ]
            keys: list[str] = [
                win._model_combo.itemData(i)
                for i in range(win._model_combo.count())
            ]
            assert labels == [
                "Lateral three-layer",
                "Three-layer",
                "HBV",
                "HBV lateral",
            ]
            assert keys == [
                "lateral_three_layer",
                "three_layer",
                "hbv",
                "hbv_lateral",
            ]
        finally:
            _close_window(win)

    def test_switching_model_resets_panel_and_clears_results(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A switch reloads the panel, clears the figures, and re-titles."""
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
            win._auto_run_checkbox.setChecked(False)
            win._model_combo.setCurrentIndex(
                win._model_combo.findData("hbv")
            )
            _pump()

            assert win._input_data.spec.key == "hbv"
            values: dict[str, float] = win._param_panel.param_values()
            assert "shallow.perc" in values
            assert "snow_hs.tt" not in values
            assert win._results is None
            assert win._figure_panel.page_count() == 0
            assert win._objective_model._table.empty
            assert win.windowTitle() == "East River calibration — HBV"
            assert win._status_label.text() == "Switched to HBV."
        finally:
            _close_window(win)

    def test_full_run_after_switch_plots_every_page(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A full three-layer run after a switch plots all eight pages.

        The zone-concentration figure must skip the derived river
        species (DIC carries no per-zone column) instead of raising.
        """
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
            win._model_combo.setCurrentIndex(
                win._model_combo.findData("three_layer")
            )
            _wait_until(
                lambda: win._input_data.spec.key == "three_layer"
                and win._worker is None
                and win._results is not None
            )
            _pump(50)

            assert win._figure_panel.page_count() == 8
            assert win._figure_panel._titles[6] == "Zone concentrations"
        finally:
            _close_window(win)

    def test_switch_filters_history_to_current_model(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        model_inputs: dict[str, gui_data.GuiInputData],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only the current model's saved runs remain in the history."""
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
            hbv_input: gui_data.GuiInputData = model_inputs["hbv"]
            _write_saved_run(tmp_path, input_data)
            _write_saved_run(
                tmp_path,
                hbv_input,
                state=_full_state(hbv_input, model_key="hbv"),
            )

            win._auto_run_checkbox.setChecked(False)
            win._model_combo.setCurrentIndex(
                win._model_combo.findData("hbv")
            )
            _pump()

            assert win._history_items
            assert all(
                item.model_key == "hbv" for item in win._history_items
            )
            combo = win._param_panel._history_combo
            labels: list[str] = [
                combo.itemText(i) for i in range(combo.count())
            ]
            assert labels == [item.label for item in win._history_items]
            assert all("lateral" not in label.lower() for label in labels)
        finally:
            _close_window(win)

    def test_gradient_outcome_from_another_model_is_discarded(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A gradient outcome from another model never reaches the panel."""
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
            outcome = gui_window._GradientOutcome(
                error=None,
                table=pd.DataFrame({"x": [1.0]}),
                base_results=None,
                base_x=win._current_x(),
                mode=MODE_FULL,
                csv_path=tmp_path / "gradient.csv",
                elapsed_seconds=1.0,
                model_key="three_layer",
            )
            win._on_finished_gradient(outcome)

            assert win._gradient_state is None
            assert win._gradients_button.isEnabled()
            assert win._line_search_button.isEnabled()
        finally:
            _close_window(win)

    def test_saves_the_selected_model(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Saving under HBV writes an HBV model and records the key."""
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
            win._auto_run_checkbox.setChecked(False)
            win._model_combo.setCurrentIndex(
                win._model_combo.findData("hbv")
            )
            _pump()
            # Run once so there are results to save for the new model.
            win._start_run(win._current_x())
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )

            run_dir: Path = tmp_path / "results" / "run_hbv"
            saves: list[pt.Model] = []

            def fake_save(
                model: pt.Model,
                hydro: pt.HydroModelResults,
                rt: pt.RtModelResults,
                results_dir: Path | None = None,
            ) -> Path:
                run_dir.mkdir(parents=True, exist_ok=True)
                saves.append(model)
                return run_dir

            monkeypatch.setattr(gui_window, "save_model_run", fake_save)
            win._on_save_run()

            assert len(saves) == 1
            assert isinstance(saves[0], pt.HbvModel)
            saved = read_gui_state(run_dir)
            assert saved is not None
            assert saved.model_key == "hbv"
        finally:
            _close_window(win)


# Pinned W1 finding line (Task 2's implementation notes).
_W1_LINE: str = (
    "W1: no H+ primary species: the speciation solve has "
    "no charge-balance row"
)


class TestNetworkEditorDialog:
    """
    The T9 modal editor dialog: seeding and preset detection, the
    validation bar, the checkbox-driven remove/re-add flow, Apply
    and the resulting fingerprint, and the preset re-seed
    confirmation (the window's Yes/No idiom, faked with
    `_FakeMessageBox`).
    """

    @staticmethod
    def _make_dialog(
        network: pt.ReactionNetwork,
    ) -> gui_network_editor.NetworkEditorDialog:
        database: pt.ChemicalDatabase = pt.ChemicalDatabase.load_default()
        return gui_network_editor.NetworkEditorDialog(network, database)

    @staticmethod
    def _find_item(
        list_widget: QListWidget, name: str
    ) -> QListWidgetItem:
        for index in range(list_widget.count()):
            item: QListWidgetItem | None = list_widget.item(index)
            assert item is not None
            if item.text() == name:
                return item
        pytest.fail(f"no item named {name!r} in the browser list")

    def test_seed_preset_and_ok_bar(self, qapp: QApplication) -> None:
        seed: pt.ReactionNetwork = get_soc_calcite_network()
        dialog: gui_network_editor.NetworkEditorDialog = (
            self._make_dialog(seed)
        )
        try:
            assert dialog.network is None
            assert dialog._preset_combo.currentData() == "soc_calcite"
            assert not dialog._ok_label.isHidden()
            assert dialog._ok_label.text() == (
                "OK — 11 species, 3 minerals, 1 exchange"
            )
            assert dialog._error_label.isHidden()
            assert dialog._warning_label.isHidden()
            assert dialog._apply_button.isEnabled()
            assert dialog._save_as_button.isEnabled()
        finally:
            dialog.deleteLater()
            _pump()

    def test_unchecking_hplus_flags_and_recovering(
        self, qapp: QApplication
    ) -> None:
        seed: pt.ReactionNetwork = get_soc_calcite_network()
        dialog: gui_network_editor.NetworkEditorDialog = (
            self._make_dialog(seed)
        )
        try:
            primary_list: QListWidget = dialog._browser_list["primary"]
            item: QListWidgetItem = self._find_item(primary_list, "H+")
            primary_list.setCurrentItem(item)
            item.setCheckState(Qt.CheckState.Unchecked)
            assert dialog._dirty
            assert not dialog._apply_button.isEnabled()
            assert not dialog._save_as_button.isEnabled()
            assert dialog._ok_label.isHidden()
            assert dialog._error_label.text() == (
                "V3: stoichiometry of secondary species 'CO2(aq)' "
                "references unknown species 'H+'\n"
                "V3: stoichiometry of secondary species 'CO3--' "
                "references unknown species 'H+'"
            )
            assert dialog._warning_label.text() == _W1_LINE
            # The rebuild replaced the list's items: look the (now
            # unchecked) H+ row up again before re-checking it.
            item = self._find_item(primary_list, "H+")
            primary_list.setCurrentItem(item)
            item.setCheckState(Qt.CheckState.Checked)
            assert dialog._dirty
            assert dialog._apply_button.isEnabled()
            assert dialog._ok_label.text() == (
                "OK — 11 species, 3 minerals, 1 exchange"
            )
            assert dialog._error_label.isHidden()
            assert dialog._warning_label.isHidden()
        finally:
            dialog.deleteLater()
            _pump()

    def test_apply_accepts_and_preserves_the_fingerprint(
        self, qapp: QApplication
    ) -> None:
        seed: pt.ReactionNetwork = get_soc_calcite_network()
        dialog: gui_network_editor.NetworkEditorDialog = (
            self._make_dialog(seed)
        )
        try:
            dialog._apply_button.click()
            assert dialog.result() == QDialog.DialogCode.Accepted
            assert not dialog._dirty
            applied: pt.ReactionNetwork | None = dialog.network
            assert applied is not None
            assert (
                gui_data.fingerprint_network(applied)
                == gui_data.fingerprint_network(seed)
            )
        finally:
            dialog.deleteLater()
            _pump()

    def test_preset_reseed_confirms_or_restores(
        self,
        qapp: QApplication,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seed: pt.ReactionNetwork = get_soc_calcite_network()
        dialog: gui_network_editor.NetworkEditorDialog = (
            self._make_dialog(seed)
        )
        monkeypatch.setattr(
            gui_network_editor, "QMessageBox", _FakeMessageBox
        )
        try:
            _FakeMessageBox.answer = _FakeMessageBox.StandardButton.Yes
            dialog._dirty = True
            dialog._preset_combo.setCurrentIndex(
                dialog._preset_combo.findData("carbon")
            )
            assert not dialog._dirty
            assert dialog._preset_combo.currentData() == "carbon"
            assert dialog._ok_label.text() == (
                "OK — 8 species, 1 minerals, 1 exchange"
            )
            _FakeMessageBox.answer = _FakeMessageBox.StandardButton.No
            dialog._dirty = True
            dialog._preset_combo.setCurrentIndex(
                dialog._preset_combo.findData("chloride")
            )
            assert dialog._dirty
            assert dialog._preset_combo.currentData() == "carbon"
        finally:
            _FakeMessageBox.answer = _FakeMessageBox.StandardButton.No
            dialog.deleteLater()
            _pump()


class TestNetworkSelection:
    """
    The window's network selector (Task 10): the preset list, a
    network switch, the load / save / edit buttons, the network-
    aware history filtering, and the network persistence of saved
    runs.
    """

    def test_network_combo_lists_the_presets(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The selector offers the five presets, in order, on the default."""
        calls: list[np.ndarray] = []
        win = _make_window(
            input_data,
            tmp_path,
            monkeypatch,
            _fake_results_factory(calls, input_data),
        )
        try:
            labels: list[str] = [
                win._network_combo.itemText(i)
                for i in range(win._network_combo.count())
            ]
            assert labels == [
                spec.display_name
                for spec in network_specs.NETWORK_SPECS.values()
            ]
            keys: list[object] = [
                win._network_combo.itemData(i)
                for i in range(win._network_combo.count())
            ]
            assert keys == list(network_specs.NETWORK_SPECS)
            assert win._network_combo.currentData() == (
                network_specs.DEFAULT_NETWORK_KEY
            )
            assert win._custom_network_name is None
        finally:
            _close_window(win)

    def test_switching_to_carbon_resets_the_window(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A switch keeps the model, swaps the network, and clears state."""
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
            win._auto_run_checkbox.setChecked(False)
            win._network_combo.setCurrentIndex(
                win._network_combo.findData("carbon")
            )
            _pump()

            assert win._input_data.spec.key == "lateral_three_layer"
            carbon_fingerprint: str = gui_data.fingerprint_network(
                get_carbon_network()
            )
            assert gui_data.fingerprint_network(win._network) == (
                carbon_fingerprint
            )
            assert win._network_combo.currentData() == "carbon"
            assert win._custom_network_name is None
            assert win._results is None
            assert win._figure_panel.page_count() == 0
            assert win._objective_model._table.empty
            assert win._status_label.text() == (
                "Switched to network Carbon (SOC, no calcite)."
            )
        finally:
            _close_window(win)

    def test_loading_a_preset_matching_file_selects_the_preset(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A file carrying the carbon preset's network selects it by key."""
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
            win._auto_run_checkbox.setChecked(False)

            path: Path = tmp_path / "carbon.yaml"
            pt.NetworkFile.from_network(get_carbon_network()).to_file(
                str(path)
            )

            def fake_open(
                parent: object,
                title: str,
                directory: str,
                flt: str,
            ) -> tuple[str, str]:
                return (str(path), "")

            monkeypatch.setattr(
                gui_window.QFileDialog, "getOpenFileName", fake_open
            )
            win._on_load_network()
            _pump()

            carbon_fingerprint: str = gui_data.fingerprint_network(
                get_carbon_network()
            )
            assert gui_data.fingerprint_network(win._network) == (
                carbon_fingerprint
            )
            assert win._network_combo.currentData() == "carbon"
            assert win._custom_network_name is None
        finally:
            _close_window(win)

    def test_loading_an_invalid_file_keeps_the_network(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A file that fails to load or validate is reported and ignored."""
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
            win._auto_run_checkbox.setChecked(False)
            before: str = gui_data.fingerprint_network(win._network)

            path: Path = tmp_path / "broken.yaml"
            path.write_text(":::")

            def fake_open(
                parent: object,
                title: str,
                directory: str,
                flt: str,
            ) -> tuple[str, str]:
                return (str(path), "")

            monkeypatch.setattr(
                gui_window.QFileDialog, "getOpenFileName", fake_open
            )
            monkeypatch.setattr(
                gui_window, "QMessageBox", _FakeMessageBox
            )
            _FakeMessageBox.critical_calls.clear()
            win._on_load_network()
            _pump()

            assert len(_FakeMessageBox.critical_calls) == 1
            assert _FakeMessageBox.critical_calls[0][0] == "Load network"
            assert gui_data.fingerprint_network(win._network) == before
            assert win._network_combo.currentData() == (
                network_specs.DEFAULT_NETWORK_KEY
            )
        finally:
            _close_window(win)

    def test_loading_a_custom_file_adds_a_custom_entry(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-preset file becomes the selector's `custom: <stem>` entry."""
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
            win._auto_run_checkbox.setChecked(False)

            carbon: pt.ReactionNetwork = get_carbon_network()
            # The carbon preset without its adsorption exchange:
            # valid, and the fingerprint of no preset.
            custom: pt.ReactionNetwork = pt.ReactionNetwork(
                primary_aqueous=list(carbon.primary_aqueous),
                mineral=carbon.mineral,
                secondary=carbon.secondary,
                mineral_kinetics=carbon.mineral_kinetics,
                exchange_species=[],
            )
            fingerprint: str = gui_data.fingerprint_network(custom)
            assert fingerprint not in (
                gui_network_editor._preset_fingerprint_map()
            )

            path: Path = tmp_path / "my_net.yaml"
            pt.NetworkFile.from_network(custom).to_file(str(path))

            def fake_open(
                parent: object,
                title: str,
                directory: str,
                flt: str,
            ) -> tuple[str, str]:
                return (str(path), "")

            monkeypatch.setattr(
                gui_window.QFileDialog, "getOpenFileName", fake_open
            )
            win._on_load_network()
            _pump()

            assert gui_data.fingerprint_network(win._network) == (
                fingerprint
            )
            assert win._network_combo.currentData() == fingerprint
            assert win._custom_network_name == "my_net"
            assert win._network_combo.itemText(
                win._network_combo.findData(fingerprint)
            ) == "custom: my_net"
        finally:
            _close_window(win)

    def test_edit_cancel_keeps_the_network(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancelling the editor dialog leaves the window untouched."""
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
            win._auto_run_checkbox.setChecked(False)
            before: str = gui_data.fingerprint_network(win._network)

            def fake_edit(
                network: pt.ReactionNetwork,
                database: pt.ChemicalDatabase,
                parent: QWidget | None,
            ) -> pt.ReactionNetwork | None:
                return None

            monkeypatch.setattr(gui_window, "edit_network", fake_edit)
            win._on_edit_network()
            _pump()

            assert gui_data.fingerprint_network(win._network) == before
            assert win._network_combo.currentData() == (
                network_specs.DEFAULT_NETWORK_KEY
            )
            assert win._custom_network_name is None
        finally:
            _close_window(win)

    def test_edit_apply_adds_an_edited_entry(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Applying an edit of a preset gets the `custom: edited` entry."""
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
            win._auto_run_checkbox.setChecked(False)

            carbon: pt.ReactionNetwork = get_carbon_network()
            edited: pt.ReactionNetwork = pt.ReactionNetwork(
                primary_aqueous=list(carbon.primary_aqueous),
                mineral=carbon.mineral,
                secondary=carbon.secondary,
                mineral_kinetics=carbon.mineral_kinetics,
                exchange_species=[],
            )
            fingerprint: str = gui_data.fingerprint_network(edited)

            def fake_edit(
                network: pt.ReactionNetwork,
                database: pt.ChemicalDatabase,
                parent: QWidget | None,
            ) -> pt.ReactionNetwork | None:
                return edited

            monkeypatch.setattr(gui_window, "edit_network", fake_edit)
            win._on_edit_network()
            _pump()

            assert gui_data.fingerprint_network(win._network) == (
                fingerprint
            )
            assert win._network_combo.currentData() == fingerprint
            assert win._custom_network_name == "edited"
            assert win._network_combo.itemText(
                win._network_combo.findData(fingerprint)
            ) == "custom: edited"
        finally:
            _close_window(win)

    def test_run_matches_network(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Old runs match only the default fingerprint; new runs by equality."""
        default_fingerprint: str = gui_data.fingerprint_network(
            network_specs.get_network_spec(
                network_specs.DEFAULT_NETWORK_KEY
            ).builder()
        )
        carbon_fingerprint: str = gui_data.fingerprint_network(
            get_carbon_network()
        )

        def fake_fingerprint(run_dir: Path) -> str | None:
            return {"run_b": "abc123"}.get(run_dir.name)

        monkeypatch.setattr(
            gui_window, "saved_run_network_fingerprint", fake_fingerprint
        )

        run_a: Path = Path("run_a")
        run_b: Path = Path("run_b")
        assert gui_window._run_matches_network(run_a, default_fingerprint)
        assert not gui_window._run_matches_network(run_a, carbon_fingerprint)
        assert gui_window._run_matches_network(run_b, "abc123")
        assert not gui_window._run_matches_network(run_b, default_fingerprint)
        assert not gui_window._run_matches_network(run_b, carbon_fingerprint)

    def test_switch_filters_history_by_model_and_network(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The history list keeps only runs of the same model and network."""
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
            win._auto_run_checkbox.setChecked(False)

            default_fingerprint: str = gui_data.fingerprint_network(
                win._network
            )
            carbon_fingerprint: str = gui_data.fingerprint_network(
                get_carbon_network()
            )
            keep: Path = tmp_path / "run_keep"
            other_network: Path = tmp_path / "run_other_network"
            other_model: Path = tmp_path / "run_other_model"
            old_run: Path = tmp_path / "run_old"
            models: dict[str, str] = {
                run.name: "lateral_three_layer"
                for run in (keep, other_network, old_run)
            }
            models[other_model.name] = "hbv"
            fingerprints: dict[str, str | None] = {
                keep.name: carbon_fingerprint,
                other_network.name: default_fingerprint,
                other_model.name: carbon_fingerprint,
                old_run.name: None,
            }

            def fake_list_run_dirs() -> list[Path]:
                return [keep, other_network, other_model, old_run]

            def fake_model_key(run_dir: Path) -> str:
                return models[run_dir.name]

            def fake_fingerprint(run_dir: Path) -> str | None:
                return fingerprints.get(run_dir.name)

            monkeypatch.setattr(
                gui_window, "list_run_dirs", fake_list_run_dirs
            )
            monkeypatch.setattr(
                gui_window, "saved_run_model_key", fake_model_key
            )
            monkeypatch.setattr(
                gui_window,
                "saved_run_network_fingerprint",
                fake_fingerprint,
            )

            win._network_combo.setCurrentIndex(
                win._network_combo.findData("carbon")
            )
            _pump()
            actual: list[str] = [
                item.run_dir.name
                for item in win._history_items
                if item.run_dir is not None
            ]
            assert actual == [keep.name]

            # Back on the default network, the old run (no recorded
            # fingerprint) counts as one of it.
            win._network_combo.setCurrentIndex(
                win._network_combo.findData(
                    network_specs.DEFAULT_NETWORK_KEY
                )
            )
            _pump()
            actual = [
                item.run_dir.name
                for item in win._history_items
                if item.run_dir is not None
            ]
            assert actual == [other_network.name, old_run.name]
        finally:
            _close_window(win)

    def test_on_save_network_writes_the_file(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save network as writes the canonical file; the window is untouched."""
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
            win._auto_run_checkbox.setChecked(False)

            target: Path = tmp_path / "saved_net.yaml"

            def fake_save_file(
                parent: object,
                title: str,
                directory: str,
                flt: str,
            ) -> tuple[str, str]:
                return (str(target), "")

            monkeypatch.setattr(
                gui_window.QFileDialog, "getSaveFileName", fake_save_file
            )
            monkeypatch.setattr(
                gui_window, "QMessageBox", _FakeMessageBox
            )
            _FakeMessageBox.information_calls.clear()
            win._on_save_network()

            assert target.is_file()
            saved: pt.ReactionNetwork = pt.ReactionNetwork.from_file(
                str(target)
            )
            saved_fingerprint: str = gui_data.fingerprint_network(saved)
            assert saved_fingerprint == gui_data.fingerprint_network(
                win._network
            )
            assert _FakeMessageBox.information_calls == [
                ("Save network", f"Network written to {target}.")
            ]
            assert win._network_combo.currentData() == (
                network_specs.DEFAULT_NETWORK_KEY
            )
            assert win._custom_network_name is None
        finally:
            _close_window(win)

    def test_on_save_run_persists_the_network(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Saving a run writes network.yaml and the fingerprint to state."""
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
            win._auto_run_checkbox.setChecked(False)
            win._network_combo.setCurrentIndex(
                win._network_combo.findData("carbon")
            )
            _pump()
            # Run once so there are results to save for the new
            # network.
            win._start_run(win._current_x())
            _wait_until(
                lambda: win._worker is None and win._results is not None
            )

            run_dir: Path = tmp_path / "results" / "run_fake"

            def fake_save(
                model: pt.Model,
                hydro: pt.HydroModelResults,
                rt: pt.RtModelResults,
                results_dir: Path | None = None,
            ) -> Path:
                run_dir.mkdir(parents=True, exist_ok=True)
                return run_dir

            monkeypatch.setattr(gui_window, "save_model_run", fake_save)
            win._on_save_run()

            fingerprint: str = gui_data.fingerprint_network(win._network)
            carbon_fingerprint: str = gui_data.fingerprint_network(
                get_carbon_network()
            )
            assert fingerprint == carbon_fingerprint
            network_file: Path = run_dir / "network.yaml"
            assert network_file.is_file()
            saved: pt.ReactionNetwork = pt.ReactionNetwork.from_file(
                str(network_file)
            )
            assert gui_data.fingerprint_network(saved) == fingerprint
            state = read_gui_state(run_dir)
            assert state is not None
            assert state.network_fingerprint == fingerprint
        finally:
            _close_window(win)

    def test_run_network_reads_or_skips_the_file(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """_run_network reads network.yaml, or skips a missing / broken one."""
        good: Path = tmp_path / "good"
        good.mkdir()
        pt.NetworkFile.from_network(get_carbon_network()).to_file(
            str(good / "network.yaml")
        )
        loaded: pt.ReactionNetwork | None = gui_window._run_network(good)
        assert loaded is not None
        assert gui_data.fingerprint_network(loaded) == (
            gui_data.fingerprint_network(get_carbon_network())
        )

        missing: Path = tmp_path / "missing"
        missing.mkdir()
        assert gui_window._run_network(missing) is None

        broken: Path = tmp_path / "broken"
        broken.mkdir()
        (broken / "network.yaml").write_text(":::")
        assert gui_window._run_network(broken) is None
        assert "Could not load" in capsys.readouterr().err
