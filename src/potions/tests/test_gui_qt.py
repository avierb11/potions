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

import base64
import dataclasses
import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text
from PySide6.QtCore import QDate, QEvent, QPointF, QRect, QSize, Qt
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QFont,
    QKeyEvent,
    QKeySequence,
    QImage,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayoutItem,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMainWindow,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QStatusBar,
    QTabBar,
    QToolBar,
    QToolButton,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

import potions as pt
from potions.hydro_model import HydrologicalModel
from potions.reactive_transport.common_networks import (
    get_carbon_network,
    get_soc_calcite_network,
)
from potions.reactive_transport.rt_zone import WaterVolumeError

from ..gui import data as gui_data
from ..gui import data_dialog as gui_data_dialog
from ..gui import calibration as gui_calibration
from ..gui import figure_panel as gui_figure_panel
from ..gui import model_pool
from ..gui import model_specs
from ..gui import network_editor as gui_network_editor
from ..gui import network_specs
from ..gui import param_panel as gui_param_panel
from ..gui import window as gui_window
from ..gui.common import (
    HYDRO_THEME,
    METRIC_LABELS,
    METRIC_NAMES,
    MINERAL_PARAMS,
    RIVER_SPECIES,
    RT_DIM_PARAMS,
    RT_THEME,
    SPECIATION_ZONES,
    bound_decimals,
    param_label,
)
from ..gui.data import PlotSelection
from ..gui.gradients import (
    Perturbation,
    free_parameter_steps,
    gradient_matrix,
    objective_vector,
)
from ..gui.line_search import (
    expand_direction,
    feasible_t_interval,
    search_t_points,
)
from ..gui.state import (
    GuiState,
    MODE_FULL,
    MODE_HYDRO,
    RUN_HISTORY_FILENAME,
    read_gui_state,
)
from .test_gui import (
    alpha,
    full_state,
    line_artists,
    midpoint_vector,
    synthetic_results,
    write_saved_run,
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


def wait_until(predicate: Callable[[], bool], timeout: float = 15.0) -> None:
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


def pump(iterations: int = 50) -> None:
    """Run a few event-loop ticks (lets deferred deletions land)."""
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    for _ in range(iterations):
        app.processEvents()
        time.sleep(0.002)


def key_press(widget: QWidget, key: Qt.Key) -> None:
    """Deliver one synthetic key press and pump the event loop."""
    QApplication.sendEvent(
        widget, QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
    )
    pump(5)


def key_release(widget: QWidget, key: Qt.Key) -> None:
    """Deliver one synthetic key release and pump the event loop."""
    QApplication.sendEvent(
        widget, QKeyEvent(QEvent.Type.KeyRelease, key, Qt.KeyboardModifier.NoModifier)
    )
    pump(5)


def fake_results_factory(
    calls: list[np.ndarray] | None,
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
    a hydrology-only run.  With `calls=None` nothing is recorded,
    as when a fake session scores its batches in-process and the
    test only watches the startup run's calls.
    """

    def fake(
        x: np.ndarray, input_data: gui_data.GuiInputData
    ) -> gui_data.GuiRunResults:
        if calls is not None:
            calls.append(np.array(x))
        hydro, rt = synthetic_results(input_data)
        if hydro_only:
            return gui_data.GuiRunResults(hydro, None)
        return gui_data.GuiRunResults(hydro, rt)

    return fake


def make_window(
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


def close_window(win: gui_window.MainWindow) -> None:
    """Stop the window's timers and drop it."""
    win._debounce_timer.stop()
    win.deleteLater()
    pump()


def assert_restored_dir(win: gui_window.MainWindow, expected: Path) -> None:
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


def install_bounds_stub(
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

    class StubDialog:
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

    monkeypatch.setattr(gui_param_panel, "BoundsDialog", StubDialog)
    return records


def double_click_event() -> QMouseEvent:
    """A plain left-button double-click mouse event."""
    return QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        QPointF(10.0, 10.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def band_of(cell: QWidget) -> str:
    """
    The row band a grid cell's stylesheet asks for (its
    background-color).
    """
    match = re.search(r"background-color: (\S+);", cell.styleSheet())
    assert match is not None, f"No band in {cell.styleSheet()!r}"
    return match.group(1)


def app_font_12pt() -> QFont:
    """The app's 12 pt base font, as `main` sets it on the app.

    The test windows never pass through `main`, so the tests that
    compare pixel-level rendering set the font on the widgets they
    probe.
    """
    font: QFont = QFont()
    font.setPointSize(12)
    return font


def objective_test_table() -> pd.DataFrame:
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


class FakeMessageBox:
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
        return FakeMessageBox.answer

    @staticmethod
    def critical(
        parent: object | None,
        title: str,
        text: str,
        *buttons: object,
    ) -> None:
        FakeMessageBox.critical_calls.append((title, text))

    @staticmethod
    def information(
        parent: object | None,
        title: str,
        text: str,
        *buttons: object,
    ) -> None:
        FakeMessageBox.information_calls.append((title, text))


# ------------------------------------------------------------------ #
# MainWindow: startup
# ------------------------------------------------------------------ #


class TestWindowGeometryClamp:
    """`_clamped_geometry` keeps a requested geometry inside the screen."""

    def test_geometry_inside_is_untouched(self) -> None:
        assert gui_window.clamped_geometry(
            QRect(10, 20, 1100, 800), QRect(0, 0, 1920, 1168)
        ) == QRect(10, 20, 1100, 800)

    def test_oversized_geometry_is_capped(self) -> None:
        # The real bug's shape: a maximized client rect saved on a
        # 1920×1200 screen, restored on a smaller available area.
        assert gui_window.clamped_geometry(
            QRect(0, 0, 1914, 1135), QRect(0, 64, 1024, 700)
        ) == QRect(0, 64, 1024, 700)

    def test_far_edge_position_is_pulled_back(self) -> None:
        # Full-size window pushed fully inside the available area.
        assert gui_window.clamped_geometry(
            QRect(1500, 1500, 800, 600), QRect(0, 0, 1920, 1200)
        ) == QRect(1120, 600, 800, 600)

    def test_headless_platform_reports_no_screen(
        self, qapp: QApplication
    ) -> None:
        # The Qt offscreen plugin reports a synthetic 800x800 "screen"
        # that is no display to fit; the helper must call it headless
        # so windows keep their intended geometry there.
        assert qapp.platformName() == "offscreen"  # the suite's platform
        assert gui_window.available_screen_geometry() is None


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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump(50)
            # Exactly one run was started, at the default (midpoint)
            # vector, and the results are on screen.
            assert len(calls) == 1
            assert np.array_equal(calls[0], midpoint_vector(input_data))
            # The finished worker's reference was dropped and its
            # deferred deletion has landed: a later "maybe start"
            # must neither touch the deleted object nor re-run.
            win._maybe_start_run()
            assert len(calls) == 1
            assert re.fullmatch(
                r"Run 1 finished in \d+\.\d s\.", win._status_label.text()
            )
            assert win._save_action.isEnabled()
            # The run-history toggle starts off.
            assert not win._save_history_action.isChecked()
            # The results on screen correspond to the current
            # parameters, so a manual run would do nothing.
            assert not win._run_action.isEnabled()
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
            assert win.windowTitle() == gui_window.title_for(
                model_specs.DEFAULT_MODEL_SPEC
            )
        finally:
            close_window(win)

    def test_starts_with_tabbed_figure_docks(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The first run's eight figures open as tabbed dock panels."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump(50)
            win.show()
            pump()
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            assert panel.page_count() == 8
            titles: list[str] = [
                "Hydrology",
                "Storage",
                "Partitioning",
                "River concentrations",
                "River loads",
                "Concentration vs. discharge",
                "Zone concentrations",
                "Reaction rates (SOC(s))",
            ]
            docks: list[gui_figure_panel.FigureDock] = panel.docks()
            assert [dock.windowTitle() for dock in docks] == titles
            # The docks share a single tab bar, tabified in page order.
            tab_bars: list[QTabBar] = [
                bar for bar in win.findChildren(QTabBar) if bar.isVisible()
            ]
            assert len(tab_bars) == 1
            assert [
                tab_bars[0].tabText(index)
                for index in range(tab_bars[0].count())
            ] == titles
        finally:
            close_window(win)

    def test_objective_panel_fits_content_without_scrollbars(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The objective table needs no scrollbars and sits top-aligned."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump(50)
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
            width, height = gui_window.objective_panel_size(
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
                gui_window.OBJECTIVE_PANEL_MARGINS,
                gui_window.OBJECTIVE_PANEL_SPACING,
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
            close_window(win)

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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump(50)
            font: QFont = app_font_12pt()
            win._objective_table.setFont(font)
            win._objective_heading.setFont(font)
            win._objective_model.set_table(objective_test_table())
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
                > h.sectionSizeHint(c) + gui_window.CHECKBOX_INSET
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
            width, height = gui_window.objective_panel_size(
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
                gui_window.OBJECTIVE_PANEL_MARGINS,
                gui_window.OBJECTIVE_PANEL_SPACING,
            )
            assert (
                win._objective_panel.width(),
                win._objective_panel.height(),
            ) == (width, height)
        finally:
            close_window(win)

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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump(50)
            font: QFont = app_font_12pt()
            win._objective_table.setFont(font)
            win._objective_heading.setFont(font)
            win._objective_model.set_table(objective_test_table())
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
            assert first_dark >= gui_window.CHECKBOX_INSET - 2
            # The full text is visible: the ink span matches the
            # font-metrics width of the display string.
            assert abs((last_dark - first_dark) - tw) <= 8
            # The text ends where a shifted text should end: inset
            # plus the style's 2 px text margin plus the full width
            # (the ink trail ends a pixel or two short of the
            # advance width): it is not clipped at the cell edge.
            assert abs(last_dark - (gui_window.CHECKBOX_INSET + 2 + tw)) <= 10
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
            close_window(win)

    def test_restores_full_state(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A complete gui_state restores everything without a re-run."""
        run_dir: Path = write_saved_run(
            tmp_path,
            input_data,
            full_state(input_data, save_history=True),
        )
        calls: list[np.ndarray] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            gui_window,
            "arr_to_results",
            fake_results_factory(calls, input_data),
        )
        available: QRect = QRect(40, 30, 760, 560)
        monkeypatch.setattr(
            gui_window, "available_screen_geometry", lambda: available
        )
        win = gui_window.MainWindow(input_data)
        try:
            # No run was started: the saved results are used as is
            # (the saved directory is kept as the relative path the
            # lookup returned, so resolve it against the cwd).
            assert calls == []
            assert win._worker is None
            assert_restored_dir(win, run_dir)
            assert win._status_label.text() == (
                "Restored the most recent saved run; the figures were "
                "built from its saved output."
            )
            assert win._save_action.isEnabled()
            # The restored results match the restored parameters.
            assert not win._run_action.isEnabled()
            panel: gui_param_panel.ParamPanel = win._param_panel
            assert panel.param_values()["proportion.1"] == pytest.approx(0.7)
            assert panel.current_bounds() == dict(input_data.bounds)
            assert panel.selection() == gui_data.default_plot_selection(
                input_data.spec, window=input_data.window
            )
            # The save-history toggle is part of the saved state.
            assert win._save_history_action.isChecked()
            # The saved 1100×800 at (10, 20) is clamped to the
            # patched screen area: the size caps to it and the
            # origin pulls back to its corner.
            assert win.geometry() == gui_window.clamped_geometry(
                QRect(10, 20, 1100, 800),
                available,
            )
            # The eight figure pages are present (their dock layout is
            # restored in Task 4; the splitter's proportional-sizes
            # assertion retired with the splitter).
            assert win._figure_panel.page_count() == 8
        finally:
            close_window(win)

    def test_restores_maximized_run(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A run saved maximized reopens maximized, not oversized."""
        run_dir: Path = write_saved_run(
            tmp_path,
            input_data,
            full_state(input_data, window_maximized=True),
        )
        calls: list[np.ndarray] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            gui_window,
            "arr_to_results",
            fake_results_factory(calls, input_data),
        )
        available: QRect = QRect(40, 30, 760, 560)
        monkeypatch.setattr(
            gui_window, "available_screen_geometry", lambda: available
        )
        win = gui_window.MainWindow(input_data)
        try:
            # The saved normal geometry is applied clamped to the
            # patched screen area, before the window is shown.
            clamped: QRect = gui_window.clamped_geometry(
                QRect(10, 20, 1100, 800), available
            )
            assert win.geometry() == clamped
            win.show()
            pump()
            # The deferred maximize (the showEvent hand-off) landed:
            # the window is genuinely maximized, not a normal window
            # wearing a screen-sized geometry.
            assert win.isMaximized()
            # The deferred maximize did not corrupt the tracked
            # normal geometry: it holds the window's own restored
            # height (the clamped 560), while the maximized client
            # is the offscreen plugin's fake-screen area.  (The
            # width cannot be pinned here: the plugin expands the
            # shown window to the figures' minimum size hint, wider
            # than the fake screen.)
            normal: QRect | None = win._normal_geometry
            assert normal is not None
            assert normal.height() == clamped.height()
            assert win.geometry() != normal
            assert_restored_dir(win, run_dir)
            # Un-maximizing returns to the tracked normal geometry:
            # what a later save records, not the maximized client.
            win.showNormal()
            pump()
            assert not win.isMaximized()
            assert win.geometry() == normal
        finally:
            close_window(win)

    def test_restores_legacy_run(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A run saved before gui_state exists restores from CSVs only."""
        run_dir: Path = write_saved_run(tmp_path, input_data)
        calls: list[np.ndarray] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            gui_window,
            "arr_to_results",
            fake_results_factory(calls, input_data),
        )
        available: QRect = QRect(40, 30, 760, 560)
        monkeypatch.setattr(
            gui_window, "available_screen_geometry", lambda: available
        )
        win = gui_window.MainWindow(input_data)
        try:
            assert calls == []
            status: str = win._status_label.text()
            assert "parameters.csv (61 values, 4-decimal;" in status
            panel: gui_param_panel.ParamPanel = win._param_panel
            # Without a saved selection the defaults are used, and the
            # initial-state parameters fall back to their midpoints.  The
            # plotting window is the record's span: the default's later
            # end is clamped by the date editors to the record's last day.
            assert panel.selection() == gui_data.default_plot_selection(
                input_data.spec, window=input_data.window
            )
            for zone in SPECIATION_ZONES:
                key: str = f"initial_state.{zone}_x"
                lo, hi = input_data.bounds[key]
                assert panel.param_values()[key] == pytest.approx(
                    0.5 * (lo + hi)
                )
            # Geometry keeps its default (the default size clamped
            # to the patched screen area).
            assert win.width() == min(
                gui_window.INITIAL_SIZE[0], available.width()
            )
            assert win.height() == min(
                gui_window.INITIAL_SIZE[1], available.height()
            )
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
            # No saved state carried the toggle: it restores off.
            assert not win._save_history_action.isChecked()
            assert_restored_dir(win, run_dir)
        finally:
            close_window(win)

    def test_restores_hydro_only(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hydrology-only saved run restores its mode, no re-run."""
        state = full_state(input_data, mode=MODE_HYDRO)
        run_dir: Path = write_saved_run(
            tmp_path, input_data, state, hydro_only=True
        )
        calls: list[np.ndarray] = []
        hydro_calls: list[np.ndarray] = []
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            gui_window,
            "arr_to_results",
            fake_results_factory(calls, input_data),
        )
        monkeypatch.setattr(
            gui_window,
            "arr_to_hydro_results",
            fake_results_factory(hydro_calls, input_data, hydro_only=True),
        )
        win = gui_window.MainWindow(input_data)
        try:
            # The restore applies the saved results and mode without
            # running either model.
            assert calls == []
            assert hydro_calls == []
            assert win._mode_action.isChecked()
            panel: gui_param_panel.ParamPanel = win._param_panel
            assert panel.param_values()["proportion.1"] == pytest.approx(0.7)
            win.show()
            pump()
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
            # The retired header's title and page label: the first
            # dock is the active page and the current page is 0.
            docks: list[gui_figure_panel.FigureDock] = figure.docks()
            assert docks[0].windowTitle() == "Hydrology"
            assert figure.current_page() == 0
            assert win._objective_model.columnCount() == 1
            assert win._results is not None
            assert win._results.rt is None
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
            assert win._status_label.text() == (
                "Restored the most recent saved run; the figures were "
                "built from its saved output."
            )
            assert_restored_dir(win, run_dir)
        finally:
            close_window(win)


# ------------------------------------------------------------------ #
# MainWindow: menus
# ------------------------------------------------------------------ #


class TestWindowMenus:
    """The menu bar carries the window's menus and their actions."""

    def test_file_menu_carries_the_data_and_network_actions(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The File menu holds the four data/network actions, in order, with tooltips."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            texts = [
                action.text().replace("&", "")
                for action in win._file_menu.actions()
            ]
            assert texts == [
                "Load data…",
                "Load network file…",
                "Save network as…",
                "Edit network…",
            ]
            assert texts and all(
                action.toolTip() for action in win._file_menu.actions()
            )
        finally:
            close_window(win)

    def test_options_menu_carries_the_settings_actions(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The Options menu holds the settings toggles, Reset bounds, and Quit."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            texts = [
                action.text().replace("&", "")
                for action in win._options_menu.actions()
                if not action.isSeparator()
            ]
            assert texts == [
                "Auto-run",
                "Save run history",
                "Reset bounds",
                "Quit",
            ]
            assert win._auto_run_action.isCheckable()
            assert win._auto_run_action.isChecked()
            assert win._save_history_action.isCheckable()
            assert not win._save_history_action.isChecked()
            assert win._reset_action.text().replace("&", "") == (
                "Reset bounds"
            )
        finally:
            close_window(win)

    def test_menu_bar_lists_the_menus_in_order(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The menu bar carries File, Options, and View, in that order."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            assert [
                action.text() for action in win.menuBar().actions()
            ] == ["&File", "&Options", "&View"]
        finally:
            close_window(win)


# ------------------------------------------------------------------ #
# MainWindow: toolbar and status bar
# ------------------------------------------------------------------ #


class TestWindowToolbar:
    """
    The window's chrome: a fixed toolbar and a bottom status bar.

    The toolbar replaces the old header row.  It is pinned chrome
    (not movable, not floatable, text-only), and a window too narrow
    for its full set reaches the overflow actions through the
    toolbar's extension button (») instead of pinning the window's
    minimum width.
    """

    def test_toolbar_is_fixed_chrome(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The toolbar cannot be dragged, floated, or shown as icons."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            assert not win._toolbar.isMovable()
            assert not win._toolbar.isFloatable()
            assert (
                win._toolbar.toolButtonStyle()
                == Qt.ToolButtonStyle.ToolButtonTextOnly
            )
        finally:
            close_window(win)

    def test_toolbar_carries_the_combos_spinner_and_job_actions_in_order(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The toolbar's items are the combos, the job actions, and the mode toggle."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            # `actions()` also contains the `QWidgetAction` wrappers
            # of the combos and the spinner; those tell apart from
            # the real toolbar actions by type.
            labels = [
                a.text() if not a.isSeparator() else "—"
                for a in win._toolbar.actions()
                if not isinstance(a, QWidgetAction)
            ]
            # Final chrome: the settings toggles and Reset bounds
            # live in the Options menu, not the toolbar.
            assert labels == [
                "—",
                "Run model",
                "Gradients",
                "Line search",
                "Calibrate",
                "Save run",
                "—",
                "Hydrology only",
            ]
            assert win._mode_action.isCheckable()
            for widget in (win._model_combo, win._network_combo, win._job_spinner):
                assert isinstance(widget.parent(), QToolBar)
        finally:
            close_window(win)

    def test_status_label_lives_in_the_status_bar(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The status label moved from the header row to the status bar."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            assert isinstance(win._status_label.parent(), QStatusBar)
            assert win._status_label.maximumWidth() == gui_window.STATUS_MAX_WIDTH
        finally:
            close_window(win)

    def test_narrow_window_elides_the_toolbar(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A window narrower than the toolbar shows the extension button (»)."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            # Qt's private extension button (») is a child tool
            # button with this object name; it becomes visible only
            # when the toolbar must elide items.  The toolbar's
            # sizeHint is ~818 px, but the combos shrink toward
            # their `_COMBO_MIN_WIDTH` floors first, so eliding
            # starts only below ~680 px; 600 is safely past that.
            extension = win._toolbar.findChild(
                QToolButton, "qt_toolbar_ext_button"
            )
            assert extension is not None
            win.show()
            pump(50)
            win.resize(1600, 700)
            pump(50)
            assert not extension.isVisible()
            win.resize(600, 700)
            pump(50)
            assert extension.isVisible()
            win.resize(1600, 700)
            pump(50)
            assert not extension.isVisible()
        finally:
            close_window(win)


# ------------------------------------------------------------------ #
# MainWindow: minimum size
# ------------------------------------------------------------------ #


class TestWindowMinimumSize:
    """
    The window's reported minimum must fit inside its default size.

    The window manager honours `minimumSizeHint`, so chrome whose
    minimum is the sum of its items' full text widths pins the
    window wider than a screen — the window can never be shrunk
    or opened onto it.  The toolbar must instead elide under
    pressure (its extension button collecting the items that no
    longer fit, and labels clipping, their full text staying in
    tooltips and combo popups) rather than push the window's
    minimum past its own default size.
    """

    def test_minimum_hint_fits_within_the_default_size(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A default-state window reports a minimum no larger than _INITIAL_SIZE."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            minimum: QSize = win.minimumSizeHint()
            assert minimum.width() <= gui_window.INITIAL_SIZE[0]
            assert minimum.height() <= gui_window.INITIAL_SIZE[1]
        finally:
            close_window(win)


# ------------------------------------------------------------------ #
# MainWindow: the View menu and the dock layout
# ------------------------------------------------------------------ #


def figure_toggle_actions(win: gui_window.MainWindow) -> list[QAction]:
    """The *View* menu's checkable per-figure toggles, in menu order."""
    return [
        action for action in win._view_menu.actions() if action.isCheckable()
    ]


def figure_tab_bar(win: gui_window.MainWindow) -> QTabBar:
    """
    The laid-out tab bar the figure docks share.

    Exactly one visible bar sits inside the window's own rect.  A
    restored dock layout (a ``restoreState`` application) can leave
    the pre-restore area's bar orphaned: never laid out, parked at
    its default geometry far outside the window, so the window clips
    it and it is invisible to the user — the geometry filter drops
    it here for the same reason.
    """
    window_rect: QRect = QRect(0, 0, win.width(), win.height())
    tab_bars: list[QTabBar] = [
        bar
        for bar in win.findChildren(QTabBar)
        if bar.isVisible() and bar.geometry().intersects(window_rect)
    ]
    assert len(tab_bars) == 1, "Expected exactly one visible tab bar."
    return tab_bars[0]


def figure_tab_texts(win: gui_window.MainWindow) -> list[str]:
    """The figure strip's tab texts."""
    bar: QTabBar = figure_tab_bar(win)
    return [bar.tabText(index) for index in range(bar.count())]


def visible_tab(win: gui_window.MainWindow) -> tuple[int, str]:
    """
    The figure strip's raised tab as its (index, title).

    The panel's `current_page()` is its own tracker, which a dock's
    `visibilityChanged` handler rewrites whenever Qt's dock layout
    raises that dock; the laid-out tab bar's current index is what
    the user actually sees, so tests about which figure is on screen
    read it here.
    """
    bar: QTabBar = figure_tab_bar(win)
    index: int = bar.currentIndex()
    return index, bar.tabText(index)


class TestViewMenu:
    """
    The *View* menu's per-figure toggles and *Reset dock layout*.

    The menu mirrors the figure docks' open state: each open dock has
    a checked, checkable action, closing a dock with its title-bar
    button unchecks its action, and re-checking the action shows the
    dock and its tab again.  *Reset dock layout* re-tabs floated and
    moved docks into the default strip at the window's top, page
    order intact.
    """

    def test_view_menu_lists_the_current_figures(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One checked, checkable toggle per figure page, in page order."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            assert win._view_menu.title() == "&View"
            toggles: list[QAction] = figure_toggle_actions(win)
            assert [action.text() for action in toggles] == panel._titles
            assert len(toggles) == panel.page_count() == 8
            assert all(
                action.isCheckable() and action.isChecked()
                for action in toggles
            )
            static: list[QAction] = [
                action
                for action in win._view_menu.actions()
                if not action.isSeparator() and not action.isCheckable()
            ]
            assert [action.text() for action in static] == ["Reset dock layout"]
            assert static[0] is win._reset_dock_action
            assert not win._reset_dock_action.isCheckable()
            assert win._reset_dock_action.toolTip()
        finally:
            close_window(win)

    def test_closing_a_dock_unchecks_its_action(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        The title-bar close and the menu toggle drive each other.

        Closing the raised dock (the ✕ button's path: the
        `closeEvent` override emits `closed_from_layout`) unchecks
        the action and drops the dock's tab; re-checking the action
        shows the dock and puts the tab back.
        """
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win.show()
            pump()
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            dock: gui_figure_panel.FigureDock = panel.docks()[0]
            action: QAction = figure_toggle_actions(win)[0]
            assert action.text() == "Hydrology"
            assert action.isChecked()
            dock.close()
            pump()
            assert not action.isChecked()
            assert not dock.isVisible()
            assert figure_tab_texts(win) == panel._titles[1:]
            action.setChecked(True)
            pump()
            assert action.isChecked()
            assert dock.isVisible()
            assert figure_tab_texts(win) == panel._titles
        finally:
            close_window(win)

    def test_reset_dock_layout_retabs_everything(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Floated docks return to one page-ordered tab strip."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win.show()
            pump()
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            panel.docks()[1].setFloating(True)
            panel.docks()[4].setFloating(True)
            pump()
            win._reset_dock_action.trigger()
            pump()
            docks: list[gui_figure_panel.FigureDock] = panel.docks()
            assert all(not dock.isFloating() for dock in docks)
            assert all(
                win.dockWidgetArea(dock) == gui_figure_panel.DOCK_AREA
                for dock in docks
            )
            assert figure_tab_texts(win) == panel._titles
            assert panel.current_page() == 0
            assert win._status_label.text() == "Dock layout reset."
        finally:
            close_window(win)

    def test_view_menu_tracks_mode_switch(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Switching to hydrology-only re-lists the menu for three figures."""
        full_calls: list[np.ndarray] = []
        hydro_calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(full_calls, input_data),
        )
        monkeypatch.setattr(
            gui_window,
            "arr_to_hydro_results",
            fake_results_factory(hydro_calls, input_data, hydro_only=True),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win.show()
            pump()
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            assert len(figure_toggle_actions(win)) == 8
            win._mode_action.setChecked(True)
            wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            pump(50)
            toggles: list[QAction] = figure_toggle_actions(win)
            assert [action.text() for action in toggles] == [
                "Hydrology",
                "Storage",
                "Partitioning",
            ]
            assert panel._titles == [
                "Hydrology",
                "Storage",
                "Partitioning",
            ]
            assert all(action.isCheckable() and action.isChecked() for action in toggles)
        finally:
            close_window(win)


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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump(50)
            win._maybe_start_run()  # nothing to do: the deleted-worker case
            assert len(calls) == 1
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
            row._spin.setValue(0.775)
            wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            assert len(calls) == 2
            assert calls[1][index] == pytest.approx(0.775)
            assert re.fullmatch(
                r"Run 2 finished in \d+\.\d s\.", win._status_label.text()
            )
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
        finally:
            close_window(win)

    def test_run_history_toggle(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The run-history toggle gates the abbreviated history file."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            history_file: Path = (
                tmp_path / "results" / RUN_HISTORY_FILENAME
            )
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            # The startup run finished with the toggle off: no
            # history file.
            assert not history_file.is_file()

            win._save_history_action.setChecked(True)
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
            row._spin.setValue(0.775)
            wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            pump(50)
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
            win._save_history_action.setChecked(False)
            row._spin.setValue(0.825)
            wait_until(
                lambda: win._run_generation == 3 and win._worker is None
            )
            pump(50)
            lines = history_file.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 1
        finally:
            close_window(win)

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
            wait_until(
                lambda: win._worker is None
                and win._status_label.text() == "Run failed: ValueError: boom"
            )
            assert win._results is None
            assert win._last_run_x is None
            assert not win._save_action.isEnabled()
            # No valid results on screen: a manual run can retry.
            assert win._run_action.isEnabled()
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
            close_window(win)

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

        monkeypatch.setattr(gui_window, "QMessageBox", FakeMessageBox)
        FakeMessageBox.critical_calls.clear()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", fake)
        win = gui_window.MainWindow(input_data)
        try:
            wait_until(
                lambda: win._worker is None
                and win._status_label.text()
                == f"Run failed: WaterVolumeError: {message}"
            )
            pump()
            assert FakeMessageBox.critical_calls == [
                ("Water volume error", message)
            ]
            assert win._results is None
            assert not win._save_action.isEnabled()
            assert win._run_action.isEnabled()
        finally:
            close_window(win)

    def test_save_run(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save writes the run output plus the session state."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
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

            # Show the window first: the save records the client
            # geometry Qt has actually applied, which only settles
            # once the window is laid out against a screen.
            win.show()
            pump()
            # Showing an offscreen window moves it off the origin;
            # restore the intended client geometry so the saved state
            # is exactly the geometry the constructor asked for.
            # Qt can still apply a different size, so record the
            # geometry it actually applied.
            win.setGeometry(0, 0, *gui_window.INITIAL_SIZE)
            pump()
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
            assert saved.selection == gui_data.default_plot_selection(
                input_data.spec, window=input_data.window
            )
            # The window was shown and restored to the default origin
            # and size, so it saves exactly that client geometry.
            assert saved.window_geometry == expected_geometry
            # The window is a normal window here: the flag is off.
            assert saved.window_maximized is False
            # The main splitter is retired: version 9 writes null
            # there (the dock layout arrives with `dock_state`).
            assert saved.splitter_sizes is None

            # The run-history toggle is part of the saved state:
            # off here by default, and the second save records it on.
            assert saved.save_history is False
            win._save_history_action.setChecked(True)
            win._on_save_run()
            assert len(saves) == 2
            saved_on = read_gui_state(run_dir)
            assert saved_on is not None
            assert saved_on.save_history is True

            # A parameter change after the run blocks the save (the
            # results on screen would be stale).
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
            row._spin.setValue(0.775)
            win._debounce_timer.stop()
            win._on_save_run()
            assert win._status_label.text() == (
                "The model is still catching up; let the current run "
                "finish before saving."
            )
            assert len(saves) == 2
        finally:
            close_window(win)

    def test_save_run_while_maximized(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A maximized save records the flag and the normal geometry."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
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
                saves.append({"results_dir": results_dir})
                return run_dir

            win.show()
            pump()
            # Record the geometry Qt actually applied after show (the
            # offscreen plugin offsets the origin by its fake frame and
            # expands the window to its content's minimum size hint,
            # which on this platform is wider than the fake screen —
            # so maximizing *shrinks* it here; that is fine, what the
            # save must record is the tracked normal geometry, never
            # the maximized client rect).
            win.setGeometry(10, 10, 500, 400)
            pump()
            normal: QRect = win.geometry()
            win.showMaximized()
            pump()
            assert win.isMaximized()
            assert win.geometry() != normal  # sanity: truly maximized

            monkeypatch.setattr(gui_window, "save_model_run", fake_save)
            win._on_save_run()
            assert len(saves) == 1
            saved = read_gui_state(run_dir)
            assert saved is not None
            assert saved.window_maximized is True
            assert saved.window_geometry == {
                "x": normal.x(),
                "y": normal.y(),
                "width": normal.width(),
                "height": normal.height(),
            }
        finally:
            close_window(win)

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
            x: np.ndarray, input_data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            release: threading.Event = threading.Event()
            releases.append(release)
            release.wait()
            return gui_data.GuiRunResults(*synthetic_results(input_data))

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", blocking_fake)
        monkeypatch.setattr(gui_window, "QMessageBox", FakeMessageBox)
        win = gui_window.MainWindow(input_data)
        try:
            # First run in flight: declining cancels the close.
            wait_until(
                lambda: win._worker is not None and win._worker.isRunning()
            )
            FakeMessageBox.answer = FakeMessageBox.StandardButton.No
            event: QCloseEvent = QCloseEvent()
            win.closeEvent(event)
            assert not event.isAccepted()
            assert win._worker is not None and win._worker.isRunning()

            # Let the run finish: the results come up as usual.
            releases[0].set()
            wait_until(
                lambda: win._worker is None
                and "Run 1 finished" in win._status_label.text()
            )

            # Second run in flight: accepting waits it out.
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
            row._spin.setValue(0.775)
            wait_until(
                lambda: win._run_generation == 2 and len(releases) == 2
            )
            threading.Timer(0.3, releases[1].set).start()
            FakeMessageBox.answer = FakeMessageBox.StandardButton.Yes
            event2: QCloseEvent = QCloseEvent()
            win.closeEvent(event2)
            assert event2.isAccepted()
            wait_until(
                lambda: win._worker is None
                and win._results is not None
            )
            assert win._run_generation == 2
        finally:
            close_window(win)

    def test_arrow_actions_flip_figure_pages(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The Left / Right actions cycle through the three figures."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            # The window now carries menu and toolbar actions too;
            # the arrow-key shortcuts single out the two figure-page
            # actions from them.  (`matches` alone is not a filter:
            # every key is a `PartialMatch` of an empty sequence, so
            # only an `ExactMatch` counts.)
            exact_left = QKeySequence.SequenceMatch.ExactMatch
            actions: dict[str, QAction] = {
                action.text(): action
                for action in win.findChildren(QAction)
                if action.shortcut().matches(Qt.Key.Key_Left) == exact_left
                or action.shortcut().matches(Qt.Key.Key_Right) == exact_left
            }
            assert set(actions) == {"Previous figure", "Next figure"}
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            docks: list[gui_figure_panel.FigureDock] = panel.docks()
            assert panel.current_page() == 0
            actions["Next figure"].triggered.emit()
            assert panel.current_page() == 1
            assert docks[1].windowTitle() == "Storage"
            actions["Next figure"].triggered.emit()
            assert panel.current_page() == 2
            assert docks[2].windowTitle() == "Partitioning"
            # Wrap-around in both directions.
            for _ in range(6):
                actions["Next figure"].triggered.emit()
            assert panel.current_page() == 0
            actions["Previous figure"].triggered.emit()
            assert panel.current_page() == 7
            assert docks[7].windowTitle() == "Reaction rates (SOC(s))"
        finally:
            close_window(win)

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
            fake_results_factory(full_calls, input_data),
        )
        monkeypatch.setattr(
            gui_window,
            "arr_to_hydro_results",
            fake_results_factory(hydro_calls, input_data, hydro_only=True),
        )
        win = gui_window.MainWindow(input_data)
        try:
            # The startup run is in full mode (the checkbox is off).
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert len(full_calls) == 1
            assert hydro_calls == []
            win.show()
            pump()
            panel: gui_param_panel.ParamPanel = win._param_panel
            assert not panel._groups["reactive"].isHidden()
            assert not panel._groups["initial"].isHidden()
            assert win._figure_panel.page_count() == 8
            assert win._objective_model.columnCount() == 4

            # Switching to hydrology-only re-runs in that mode and
            # hides the reactive-transport parts of the panel.
            win._mode_action.setChecked(True)
            wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            pump(50)
            assert len(hydro_calls) == 1
            assert len(full_calls) == 1
            assert np.array_equal(
                hydro_calls[0], midpoint_vector(input_data)
            )
            assert panel._groups["reactive"].isHidden()
            assert panel._groups["initial"].isHidden()
            # The hidden checkboxes keep their state, so the plot
            # selection is unchanged.
            assert panel.selection() == gui_data.default_plot_selection(
                input_data.spec, window=input_data.window
            )
            assert win._figure_panel.page_count() == 3
            assert win._figure_panel._titles == [
                "Hydrology",
                "Storage",
                "Partitioning",
            ]
            assert win._objective_model.columnCount() == 1
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()

            # A parameter change in hydro mode records the mode in
            # the run history (Streamflow is the only target).
            win._save_history_action.setChecked(True)
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
            row._spin.setValue(0.775)
            wait_until(
                lambda: win._run_generation == 3 and win._worker is None
            )
            pump(50)
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
            win._mode_action.setChecked(False)
            wait_until(
                lambda: win._run_generation == 4 and win._worker is None
            )
            pump(50)
            assert len(full_calls) == 2
            index: int = list(input_data.bounds).index("proportion.1")
            assert full_calls[1][index] == pytest.approx(0.775)
            assert not panel._groups["reactive"].isHidden()
            assert not panel._groups["initial"].isHidden()
            assert win._figure_panel.page_count() == 8
            assert win._objective_model.columnCount() == 4
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
        finally:
            close_window(win)


# ------------------------------------------------------------------ #
# MainWindow: the Home+Up maximize chord
# ------------------------------------------------------------------ #


class TestMaximizeChord:
    """Home+Up toggles the window's maximize; single keys do not."""

    def test_chord_toggles_maximize(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win.show()
            pump()
            normal_geometry: QRect = win.geometry()

            # Home held, then Up: maximize.
            key_press(win, Qt.Key.Key_Home)
            key_press(win, Qt.Key.Key_Up)
            key_release(win, Qt.Key.Key_Home)
            key_release(win, Qt.Key.Key_Up)
            assert win.isMaximized()

            # Up held, then Home (either order works): un-maximize,
            # back to the previous normal geometry.
            key_press(win, Qt.Key.Key_Up)
            key_press(win, Qt.Key.Key_Home)
            key_release(win, Qt.Key.Key_Up)
            key_release(win, Qt.Key.Key_Home)
            assert not win.isMaximized()
            assert win.geometry() == normal_geometry

            # A lone Home and a lone Up never maximize, even when the
            # one follows the other with the key released in between.
            key_press(win, Qt.Key.Key_Home)
            key_release(win, Qt.Key.Key_Home)
            key_press(win, Qt.Key.Key_Up)
            key_release(win, Qt.Key.Key_Up)
            assert not win.isMaximized()
        finally:
            close_window(win)

    def test_single_up_keeps_spinbox_behavior(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win.show()
            pump()
            spin: QDoubleSpinBox | None = win._param_panel.findChild(
                QDoubleSpinBox
            )
            assert spin is not None
            spin.setFocus()
            pump()

            # A lone Up still steps the spin box (the chord filter
            # only ever consumes the *second* key of the pair).
            before: float = spin.value()
            key_press(spin, Qt.Key.Key_Up)
            key_release(spin, Qt.Key.Key_Up)
            assert spin.value() == pytest.approx(before + spin.singleStep())
            assert not win.isMaximized()

            # With Home held, Up is the chord: it maximizes the window
            # and is consumed (the spin box does not step on it).  The
            # lone Home itself is not the chord and reaches the spin
            # box, which Qt moves to its minimum.
            key_press(spin, Qt.Key.Key_Home)
            value_after_home: float = spin.value()
            key_press(spin, Qt.Key.Key_Up)
            key_release(spin, Qt.Key.Key_Home)
            key_release(spin, Qt.Key.Key_Up)
            assert win.isMaximized()
            assert spin.value() == value_after_home
        finally:
            win._debounce_timer.stop()
            close_window(win)


# ------------------------------------------------------------------ #
# MainWindow: auto-run toggle
# ------------------------------------------------------------------ #


class TestAutoRunToggle:
    """The *Auto-run* toggle and the manual *Run model* action."""

    def test_run_action_manual_run_with_auto_run_off(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With auto-run off, edits wait for the *Run model* action."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump(50)
            assert win._auto_run_action.isChecked()
            assert not win._run_action.isEnabled()
            win._auto_run_action.setChecked(False)
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
            row._spin.setValue(0.775)
            # The results on screen are stale, so the run button is
            # enabled; no auto-run is armed, so nothing else happens.
            assert win._run_action.isEnabled()
            pump(200)
            assert len(calls) == 1
            assert win._run_action.isEnabled()
            win._run_action.trigger()
            # The run started synchronously; whether or not it has
            # already finished, the button stays disabled (a run is
            # in flight, and a finished run matches the current
            # values).
            assert win._run_generation == 2
            assert not win._run_action.isEnabled()
            wait_until(
                lambda: win._worker is None and win._run_generation == 2
            )
            assert len(calls) == 2
            assert calls[1][index] == pytest.approx(0.775)
            assert not win._run_action.isEnabled()
        finally:
            close_window(win)

    def test_auto_run_off_suppresses_debounced_rerun(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Turning auto-run off before a debounce fires stops the run."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump(50)
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
            # Auto-run is on, so the edit arms the debounce; turning
            # the checkbox off before it fires must still stop the
            # run (the gate is checked when the timer fires).
            row._spin.setValue(0.775)
            win._auto_run_action.setChecked(False)
            wait_until(lambda: not win._debounce_timer.isActive())
            assert len(calls) == 1
            # Re-checking auto-run and editing again starts the run
            # with the latest value as usual.
            win._auto_run_action.setChecked(True)
            row._spin.setValue(0.78)
            wait_until(
                lambda: win._worker is None and win._run_generation == 2
            )
            assert len(calls) == 2
            assert calls[1][index] == pytest.approx(0.78)
        finally:
            close_window(win)


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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump(50)
            # The startup run finished in the default full mode, so
            # the window carries the warm reactive-transport tint.
            assert win.styleSheet() == "background-color: #FBEEDB;"
            # With auto-run off, toggling the mode updates the panel
            # and the tint but starts no re-run.
            win._auto_run_action.setChecked(False)
            win._mode_action.setChecked(True)
            assert win.styleSheet() == "background-color: #EAF1F8;"
            win._mode_action.setChecked(False)
            assert win.styleSheet() == "background-color: #FBEEDB;"
            assert len(calls) == 1
        finally:
            close_window(win)


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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            combo = win._param_panel._history_combo
            # No saved run on disk: the dropdown stays empty until
            # the startup run has finished.
            assert combo.count() == 0
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert combo.count() == 1
            assert re.fullmatch(
                r"Run 1 \u2014 \d{2}:\d{2}:\d{2} \(full\)",
                combo.itemText(0),
            )

            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
            row._spin.setValue(0.775)
            wait_until(
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
            close_window(win)

    def test_restore_does_not_rerun(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Restoring a session run replaces values and results, without a re-run."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)

            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            row._spin.setValue(0.775)
            wait_until(
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
            assert isinstance(row, gui_param_panel.FreeCell)
            assert row.value() == pytest.approx(0.6)
            assert len(calls) == 2
            assert win._run_generation == 2
            last_x: np.ndarray | None = win._last_run_x
            assert last_x is not None
            assert np.array_equal(last_x, midpoint_vector(input_data))
            assert last_x[index] == pytest.approx(0.6)
            assert win._last_run_mode == MODE_FULL
            results: gui_data.GuiRunResults | None = win._results
            assert results is not None
            assert results.rt is not None
            assert not win._run_action.isEnabled()
            assert win._status_label.text() == (
                f"Restored the previous run {combo.itemText(1)}."
            )
        finally:
            close_window(win)

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
            fake_results_factory(full_calls, input_data),
        )
        monkeypatch.setattr(
            gui_window,
            "arr_to_hydro_results",
            fake_results_factory(hydro_calls, input_data, hydro_only=True),
        )
        win = gui_window.MainWindow(input_data)
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert len(full_calls) == 1

            win._mode_action.setChecked(True)
            wait_until(
                lambda: win._run_generation == 2 and win._worker is None
            )
            pump(50)
            assert len(hydro_calls) == 1

            # Restore the first (full) run.
            combo = win._param_panel._history_combo
            combo.setCurrentIndex(1)
            win._on_history_restore()
            win.show()
            pump()
            assert not win._mode_action.isChecked()
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
            pump()
            assert win._mode_action.isChecked()
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
            close_window(win)

    def test_overlay_no_rerun(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The overlay checkbox redraws the figures from the selected run, without a re-run."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert len(calls) == 1

            panel = win._param_panel
            combo = panel._history_combo
            assert combo.count() == 1
            assert combo.currentIndex() == 0

            assert not panel._history_overlay_checkbox.isChecked()
            panel._history_overlay_checkbox.setChecked(True)
            pump()

            # Page 0 is the hydrology figure: the previous run's
            # line sits behind the current one, half transparent.
            figure: Figure = win._figure_panel._canvases[0].figure
            hydro_lines = line_artists(figure.axes[0])
            assert len(hydro_lines) == 2
            assert hydro_lines[0].get_alpha() == 0.5
            assert alpha(hydro_lines[1]) == 1.0
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
            pump()
            figure = win._figure_panel._canvases[0].figure
            assert len(line_artists(figure.axes[0])) == 1
            assert len(calls) == 1
        finally:
            close_window(win)

    def test_restore_empty_history(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A restore with an empty dropdown reports that there is nothing to restore."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            # The startup run is still in flight: no history yet, and
            # a restore says exactly that.
            assert win._param_panel._history_combo.count() == 0
            win._on_history_restore()
            assert win._status_label.text() == (
                "No previous run selected: the history is empty."
            )
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            assert win._param_panel._history_combo.count() == 1
        finally:
            close_window(win)

    def test_saved_dir_restore(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A saved run directory joins the dropdown and restores from disk, without a re-run."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            index: int = list(input_data.bounds).index("proportion.1")
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)

            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            row._spin.setValue(0.775)
            wait_until(
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
            wait_until(
                lambda: win._run_generation == 3 and win._worker is None
            )
            saved_index = combo.findText(run_dir.name)
            assert saved_index == 3
            combo.setCurrentIndex(saved_index)
            win._on_history_restore()

            # The restore rebuilds the panel's cells; the fresh one
            # for this parameter carries the restored value.
            row = win._param_panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
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
            assert not win._run_action.isEnabled()
            assert win._status_label.text() == (
                f"Restored the previous run {run_dir.name}."
            )
        finally:
            close_window(win)


# ------------------------------------------------------------------ #
# FigurePanel: tabbed figure docks (standalone)
# ------------------------------------------------------------------ #


def figure(title: str) -> tuple[str, Figure]:
    """A trivial single-axis figure under the given title."""
    fig: Figure = Figure()
    fig.suptitle(title)
    fig.add_subplot()
    return title, fig


def docks_window() -> tuple[QMainWindow, QMenu]:
    """
    A bare shown main window and an empty menu for the standalone
    figure-dock tests.

    The panel fills a *View* menu with one toggle per dock; these
    tests do not inspect the toggles, so a parentless menu is enough.
    """
    window: QMainWindow = QMainWindow()
    window.resize(1200, 800)
    window.show()
    pump()
    return window, QMenu("View")


class TestFigureDocks:
    """The figure panel's one-dock-per-figure tabbing and paging."""

    def test_empty_panel(self, qapp: QApplication) -> None:
        window, view_menu = docks_window()
        try:
            panel = gui_figure_panel.FigurePanel(window, view_menu)
            assert panel.page_count() == 0
            assert panel.current_page() == -1
            assert panel.docks() == []
        finally:
            window.close()
            pump()

    def test_tabs_and_titles(self, qapp: QApplication) -> None:
        """One tabbed dock per figure, titled and named in page order."""
        window, view_menu = docks_window()
        try:
            panel = gui_figure_panel.FigurePanel(window, view_menu)
            panel.set_figures([figure("A"), figure("B"), figure("C")])
            # The dock area's tab bar fills its tabs on the next
            # event-loop pass (Qt defers the tab layout), so pump
            # before reading the tab texts.
            pump()
            docks: list[gui_figure_panel.FigureDock] = panel.docks()
            assert [dock.windowTitle() for dock in docks] == ["A", "B", "C"]
            assert [dock.objectName() for dock in docks] == [
                "figure-dock-a",
                "figure-dock-b",
                "figure-dock-c",
            ]
            # The docks share one tab bar carrying the page titles.
            tab_bars: list[QTabBar] = [
                bar for bar in window.findChildren(QTabBar) if bar.isVisible()
            ]
            assert len(tab_bars) == 1
            assert [
                tab_bars[0].tabText(index)
                for index in range(tab_bars[0].count())
            ] == ["A", "B", "C"]
            assert panel._titles == ["A", "B", "C"]
            assert panel.current_page() == 0
        finally:
            window.close()
            pump()

    def test_paging_and_wrap_around(self, qapp: QApplication) -> None:
        window, view_menu = docks_window()
        try:
            panel = gui_figure_panel.FigurePanel(window, view_menu)
            panel.set_figures(
                [
                    figure("A"),
                    figure("B"),
                    figure("C"),
                ]
            )
            assert panel.page_count() == 3
            assert panel.current_page() == 0
            assert panel.docks()[panel.current_page()].windowTitle() == "A"
            panel.next_page()
            assert panel.current_page() == 1
            assert panel.docks()[panel.current_page()].windowTitle() == "B"
            panel.next_page()
            assert panel.current_page() == 2
            assert panel.docks()[panel.current_page()].windowTitle() == "C"
            panel.next_page()  # wraps to the first page
            assert panel.current_page() == 0
            panel.previous_page()  # wraps to the last page
            assert panel.current_page() == 2
            panel.show_page(99)  # clamped to the last page
            assert panel.current_page() == 2
            panel.show_page(-3)  # clamped to the first page
            assert panel.current_page() == 0
        finally:
            window.close()
            pump()

    def test_set_figures_clamps_the_current_page(
        self, qapp: QApplication
    ) -> None:
        """Shrinking the figure list keeps the index valid."""
        window, view_menu = docks_window()
        try:
            panel = gui_figure_panel.FigurePanel(window, view_menu)
            panel.set_figures([figure("A"), figure("B"), figure("C")])
            panel.show_page(1)
            panel.set_figures([figure("A"), figure("B")])
            assert panel.current_page() == 1
            panel.set_figures([figure("A")])
            assert panel.current_page() == 0
            panel.set_figures(())
            assert panel.page_count() == 0
            assert panel.current_page() == -1
            assert panel.docks() == []
        finally:
            window.close()
            pump()

    def test_set_figures_replaces_docks(self, qapp: QApplication) -> None:
        """A new figure list retires the old docks."""
        window, view_menu = docks_window()
        try:
            panel = gui_figure_panel.FigurePanel(window, view_menu)
            panel.set_figures([figure("A"), figure("B"), figure("C")])
            old_docks: list[gui_figure_panel.FigureDock] = panel.docks()
            panel.set_figures([figure("D")])
            docks: list[gui_figure_panel.FigureDock] = panel.docks()
            assert [dock.windowTitle() for dock in docks] == ["D"]
            assert panel.current_page() == 0
            # Retirement: the old docks left the window's dock area
            # (`NoDockWidgetArea`) and the new one is docked in the
            # figure area.  The retired docks' `deleteLater` waits
            # for the event loop to return — a test never returns to
            # it the way a running application does — so the test
            # checks the dock-area membership Qt actually maintains,
            # not the deferred deletion.
            pump()
            for dock in old_docks:
                assert (
                    window.dockWidgetArea(dock)
                    == Qt.DockWidgetArea.NoDockWidgetArea
                )
            assert (
                window.dockWidgetArea(docks[0])
                == Qt.DockWidgetArea.TopDockWidgetArea
            )
        finally:
            window.close()
            pump()

    def test_save_pages(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        """Every page is saved as a PNG named after its title."""
        window, view_menu = docks_window()
        try:
            panel = gui_figure_panel.FigurePanel(window, view_menu)
            panel.set_figures(
                [
                    figure("Streamflow and river concentrations"),
                    figure("Mineral reaction rates"),
                    figure("!!!"),
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
            window.close()
            pump()

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
        window, view_menu = docks_window()
        try:
            panel = gui_figure_panel.FigurePanel(window, view_menu)
            panel.set_figures([figure("A"), figure("B")])
            for canvas in panel._canvases:
                canvas._set_device_pixel_ratio(dpr)
            for canvas in panel._canvases:
                assert canvas.figure.dpi == pytest.approx(
                    gui_figure_panel.CANVAS_DPI * dpr, rel=1e-6
                )
        finally:
            window.close()
            pump()

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
        window, view_menu = docks_window()
        try:
            panel = gui_figure_panel.FigurePanel(window, view_menu)
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
            window.close()
            pump()


# ------------------------------------------------------------------ #
# _ObjectiveTableModel (standalone)
# ------------------------------------------------------------------ #


def objective_table() -> pd.DataFrame:
    """A two-column objective table with NaNs in the second column."""
    values: dict[str, list[float]] = {
        "Streamflow": [0.9, 0.5, 0.8, 0.4, 0.01, 0.7, 0.6],
        "DOC": [float("nan")] * 7,
    }
    return pd.DataFrame(values, index=list(METRIC_LABELS.values()))


class TestObjectiveTableModel:
    """The model behind the table panel left of the parameter panel."""

    def test_empty_model(self, qapp: QApplication) -> None:
        model = gui_window.ObjectiveTableModel()
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
            pump()

    def test_set_table(self, qapp: QApplication) -> None:
        model = gui_window.ObjectiveTableModel()
        try:
            model.set_table(objective_table())
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
            pump()

    def test_cell_formats_three_visible_digits(
        self, qapp: QApplication
    ) -> None:
        """A cell shows three visible digits; scientific when large."""
        values: dict[str, list[float]] = {
            "Streamflow": [0.9, 0.876, 1234.5, 0.0000123, 0.0009, 99.9, 100.0],
        }
        model = gui_window.ObjectiveTableModel()
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
            pump()


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
            # The default's window end is clamped by the editors to the
            # record's last day, so the panel holds the record-derived
            # default.
            assert panel.selection() == gui_data.default_plot_selection(
                input_data.spec, window=input_data.window
            )
        finally:
            panel.deleteLater()
            pump()

    def test_free_cell_value(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        panel = self._make_panel(input_data)
        try:
            row = panel._rows["proportion.1"]
            assert isinstance(row, gui_param_panel.FreeCell)
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
            pump()

    def test_cell_layout(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        """Each cell places the symbol left of its value, a stretch between."""
        panel = self._make_panel(input_data)
        try:
            free = panel._rows["proportion.1"]
            assert isinstance(free, gui_param_panel.FreeCell)
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
            assert isinstance(pinned, gui_param_panel.PinnedCell)
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
            pump()

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
                band_a, band_b, line = gui_param_panel.GRID_STYLES[
                    theme.name
                ]

                cells: list[QWidget] = group.content().findChildren(
                    gui_param_panel.FreeCell
                ) + group.content().findChildren(
                    gui_param_panel.PinnedCell
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
                        assert band_of(cell) == band_of(row_cells[0])
                    row_bands.append(band_of(row_cells[0]))

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
                    gui_param_panel.ZoneHeader
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
                expected_band: str = gui_param_panel.GRID_STYLES[
                    single_group.theme.name
                ][0]
                single_cells: list[QWidget] = (
                    single_group.content().findChildren(
                        gui_param_panel.FreeCell
                    )
                    + single_group.content().findChildren(
                        gui_param_panel.PinnedCell
                    )
                )
                assert single_cells
                for single_cell in single_cells:
                    assert band_of(single_cell) == expected_band

            # The neutral group's widgets keep the default look.
            box: QCheckBox = panel._streamflow_box
            assert box.styleSheet() == ""
            assert box.objectName() == ""
        finally:
            panel.deleteLater()
            pump()

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
            assert isinstance(old, gui_param_panel.PinnedCell)
            old_band: str = band_of(old)
            install_bounds_stub(monkeypatch, (0.25, 1.75))
            qapp.sendEvent(old._value, double_click_event())
            new_cell = panel._rows[key]
            assert isinstance(new_cell, gui_param_panel.FreeCell)
            assert new_cell.objectName().startswith("grid_cell_")
            assert new_cell.testAttribute(
                Qt.WidgetAttribute.WA_StyledBackground
            )
            assert band_of(new_cell) == old_band
        finally:
            panel.deleteLater()
            pump()

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
            assert isinstance(row, gui_param_panel.PinnedCell)
            records = install_bounds_stub(monkeypatch, (0.25, 1.75))
            bounds: list[tuple[str, float, float]] = []
            values: list[str] = []
            panel.bounds_changed.connect(
                lambda k, lo, hi: bounds.append((k, lo, hi))
            )
            panel.value_changed.connect(values.append)
            qapp.sendEvent(row._value, double_click_event())
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
            assert isinstance(new_cell, gui_param_panel.FreeCell)
            assert new_cell._symbol.text() == "alpha"
            assert new_cell._symbol.toolTip() == label
            assert new_cell.value() == pytest.approx(1.0)
            assert new_cell._spin.value() == pytest.approx(1.0)
            assert new_cell._bounds == (0.25, 1.75)
            assert bounds == [(key, 0.25, 1.75)]
            assert values == [key]
        finally:
            panel.deleteLater()
            pump()

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
            assert isinstance(row, gui_param_panel.PinnedCell)
            records = install_bounds_stub(monkeypatch, None)
            bounds: list[tuple[str, float, float]] = []
            values: list[str] = []
            panel.bounds_changed.connect(
                lambda k, lo, hi: bounds.append((k, lo, hi))
            )
            panel.value_changed.connect(values.append)
            row.mouseDoubleClickEvent(double_click_event())
            assert len(records) == 1
            assert isinstance(panel._rows[key], gui_param_panel.PinnedCell)
            assert bounds == []
            assert values == []
        finally:
            panel.deleteLater()
            pump()

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
            assert isinstance(row, gui_param_panel.FreeCell)
            bounds: list[tuple[str, float, float]] = []
            values: list[str] = []
            panel.bounds_changed.connect(
                lambda k, lo, hi: bounds.append((k, lo, hi))
            )
            panel.value_changed.connect(values.append)

            records = install_bounds_stub(monkeypatch, (0.5, 0.95))
            line_edit = row._spin.lineEdit()
            assert line_edit is not None
            qapp.sendEvent(line_edit, double_click_event())
            assert len(records) == 1
            assert records[0]["title"] == (
                "Edit range — Distal proportion of catchment"
            )
            assert records[0]["hint"] == "Current range: 0.25000 to 0.95000."
            assert records[0]["lo"] == pytest.approx(0.25)
            assert records[0]["hi"] == pytest.approx(0.95)
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel.FreeCell)
            assert row._bounds == (0.5, 0.95)
            assert row.value() == pytest.approx(0.6)  # the value is kept
            assert bounds == [(key, 0.5, 0.95)]
            assert values == [key]

            # The same for the edited range.
            records = install_bounds_stub(monkeypatch, (0.5, 0.8))
            line_edit = row._spin.lineEdit()
            assert line_edit is not None
            qapp.sendEvent(line_edit, double_click_event())
            assert records[0]["hint"] == "Current range: 0.50000 to 0.95000."
            row = panel._rows[key]
            assert isinstance(row, gui_param_panel.FreeCell)
            assert row._bounds == (0.5, 0.8)
            assert bounds == [(key, 0.5, 0.95), (key, 0.5, 0.8)]
            assert values == [key, key]
        finally:
            panel.deleteLater()
            pump()

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
            assert isinstance(row, gui_param_panel.FreeCell)

            records = install_bounds_stub(monkeypatch, (0.5, 0.95))
            qapp.sendEvent(row._symbol, double_click_event())
            assert len(records) == 1
            assert records[0]["lo"] == pytest.approx(0.25)
            assert records[0]["hi"] == pytest.approx(0.95)
        finally:
            panel.deleteLater()
            pump()

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
            assert isinstance(row, gui_param_panel.FreeCell)
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
            records = install_bounds_stub(monkeypatch, (v, v))
            line_edit = row._spin.lineEdit()
            assert line_edit is not None
            qapp.sendEvent(line_edit, double_click_event())
            assert len(records) == 1
            assert records[0]["value"] == pytest.approx(v)
            assert records[0]["lo"] == pytest.approx(lo)
            assert records[0]["hi"] == pytest.approx(hi)
            pinned = panel._rows[key]
            assert isinstance(pinned, gui_param_panel.PinnedCell)
            assert panel._bounds[key] == (v, v)
            assert panel._values[key] == pytest.approx(v)
            assert bounds == [(key, v, v)]
            assert values == [key]

            # 2. A degenerate result from the pinned cell's unlock
            #    dialog is a no-op: the row stays pinned at the
            #    same value.
            records = install_bounds_stub(monkeypatch, (v, v))
            qapp.sendEvent(pinned._value, double_click_event())
            assert len(records) == 1
            assert isinstance(panel._rows[key], gui_param_panel.PinnedCell)
            assert panel._bounds[key] == (v, v)
            assert panel._values[key] == pytest.approx(v)
            assert bounds == [(key, v, v)]
            assert values == [key]

            # 3. A real range from the unlock dialog converts the
            #    cell back into a free cell, keeping the value.
            lo2: float = v - 0.2
            hi2: float = v + 0.2
            records = install_bounds_stub(monkeypatch, (lo2, hi2))
            pinned = panel._rows[key]
            qapp.sendEvent(pinned._value, double_click_event())
            assert len(records) == 1
            free_cell = panel._rows[key]
            assert isinstance(free_cell, gui_param_panel.FreeCell)
            assert panel._bounds[key] == (lo2, hi2)
            assert panel._values[key] == pytest.approx(v)
            assert bounds == [(key, v, v), (key, lo2, hi2)]
            assert values == [key, key]
        finally:
            panel.deleteLater()
            pump()

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
            pump()

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
            pump()

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
            pump()

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
            assert isinstance(row, gui_param_panel.PinnedCell)
            install_bounds_stub(monkeypatch, (0.25, 1.75))
            row.mouseDoubleClickEvent(double_click_event())
            assert isinstance(
                panel._rows[key], gui_param_panel.FreeCell
            )
            panel._values["proportion.1"] = 1.5  # outside the range
            restored: dict[str, float] = panel.reset_bounds()
            assert panel.current_bounds() == dict(input_data.bounds)
            assert restored["proportion.1"] == pytest.approx(0.95)
            assert isinstance(panel._rows[key], gui_param_panel.PinnedCell)
        finally:
            panel.deleteLater()
            pump()

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
                    == gui_param_panel.GRID_COLUMNS
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
            def assert_fitted_toggle(
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
                assert_fitted_toggle(panel._groups[name])
                assert (
                    "padding: 4px 0 4px 4px;" in panel._groups[name].styleSheet()
                )
            assert_fitted_toggle(panel._plots_group)
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
            pump()

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
                    gui_param_panel.ZoneHeader
                )
            ]
            assert any("Shallow subsurface" in text for text in labels)
            assert any("Deep subsurface" in text for text in labels)
            assert panel._groups.get("proportion") is None
        finally:
            panel.deleteLater()
            pump()

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
            pump()

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
            pump()

    def test_panel_editor_bounds(
        self, qapp: QApplication, input_data: gui_data.GuiInputData
    ) -> None:
        """The date editors span the loaded record's window, so a
        date outside the record cannot be selected."""
        window: tuple[pd.Timestamp, pd.Timestamp] = (
            pd.Timestamp("2019-01-01"),
            pd.Timestamp("2020-12-31"),
        )
        data = dataclasses.replace(input_data, window=window)
        selection = gui_data.default_plot_selection(data.spec, window=window)
        panel = gui_param_panel.ParamPanel(data, selection)
        try:
            lo: QDate = QDate.fromString("2019-01-01", Qt.DateFormat.ISODate)
            hi: QDate = QDate.fromString("2020-12-31", Qt.DateFormat.ISODate)
            assert panel._window_from_edit.minimumDate() == lo
            assert panel._window_from_edit.maximumDate() == hi
            assert panel._window_to_edit.minimumDate() == lo
            assert panel._window_to_edit.maximumDate() == hi
            # The window-derived selection round-trips through the
            # editors unchanged.
            assert panel.selection().window_start == "2019-01-01"
            assert panel.selection().window_end == "2020-12-31"
        finally:
            panel.deleteLater()
            pump()


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
        dialog = gui_param_panel.BoundsDialog(
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
            pump()


# ------------------------------------------------------------------ #
# The toolbar's *Gradients* action: the finite-difference jobs
# ------------------------------------------------------------------ #


def free_keys(input_data: gui_data.GuiInputData) -> list[str]:
    """The keys of the parameters whose loaded range has a width."""
    return [
        key for key, (low, high) in input_data.bounds.items() if high > low
    ]


def gradient_fake_factory(
    calls: list[np.ndarray] | None,
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
    free parameter.  With `calls=None` nothing is recorded, as when
    a fake session scores its batches in-process.
    """
    keys: list[str] = list(input_data.bounds)
    hydro_index: int = keys.index(hydro_term_key)
    rt_index: int = keys.index(rt_term_key)

    def make(
        x: np.ndarray,
        input_data: gui_data.GuiInputData,
        hydro_only: bool,
    ) -> gui_data.GuiRunResults:
        if calls is not None:
            calls.append(np.array(x))
        hydro, rt = synthetic_results(input_data)
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
        x: np.ndarray, input_data: gui_data.GuiInputData
    ) -> gui_data.GuiRunResults:
        return make(x, input_data, False)

    def hydro_fake(
        x: np.ndarray, input_data: gui_data.GuiInputData
    ) -> gui_data.GuiRunResults:
        return make(x, input_data, True)

    return full_fake, hydro_fake


def mode_evaluate(
    full_fake: Callable[
        [np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults
    ],
    hydro_fake: Callable[
        [np.ndarray, gui_data.GuiInputData], gui_data.GuiRunResults
    ],
    input_data: gui_data.GuiInputData,
) -> Callable[[np.ndarray, str], gui_data.GuiRunResults]:
    """
    Combine the two conversion stand-ins into one session
    evaluator: the run-mode string selects which fake scores the
    point, exactly as a warm session picks its conversion.
    """

    def evaluate(x: np.ndarray, mode: str) -> gui_data.GuiRunResults:
        if mode == MODE_HYDRO:
            return hydro_fake(x, input_data)
        return full_fake(x, input_data)

    return evaluate


class FakeRunSession:
    """
    A stand-in for ``proc_pool.ModelRunSession`` that scores each
    batch in this process.

    A real session's forkserver children re-import the ``data``
    module fresh, so a parent-side patch of the ``arr_to_*``
    conversions is invisible to them; the job tests replace the
    session class instead, and this fake maps each point through
    `evaluate`.  When `gate_path` is given each batch blocks
    (polling briefly, with a generous timeout) until the file
    exists, so the test can connect to the worker's signals before
    any point completes.  Each batch's points are recorded on
    `batches`, a point whose evaluation raises becomes the
    ``"Type: message"`` error string a real session reports, and
    `on_result` fires in input order as each point "completes".
    """

    def __init__(
        self,
        csv_path: str,
        *,
        evaluate: Callable[[np.ndarray, str], gui_data.GuiRunResults],
        gate_path: Path | None = None,
    ) -> None:
        self.csv_path: str = csv_path
        self.batches: list[list[np.ndarray]] = []
        self.closed: bool = False
        self._evaluate: Callable[
            [np.ndarray, str], gui_data.GuiRunResults
        ] = evaluate
        self._gate_path: Path | None = gate_path

    def map(
        self,
        points: Sequence[np.ndarray],
        mode: str,
        on_result: (
            Callable[[int, gui_data.GuiRunResults | str], None] | None
        ) = None,
    ) -> list[gui_data.GuiRunResults | str]:
        """
        "Run" one batch: gate, evaluate every point in-process,
        report each through `on_result`, and return the payloads in
        input order.
        """
        batch: list[np.ndarray] = [np.array(point) for point in points]
        self.batches.append(batch)
        if self._gate_path is not None:
            deadline: float = time.monotonic() + 60.0
            while (
                not self._gate_path.exists() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
        payloads: list[gui_data.GuiRunResults | str] = []
        for index, point in enumerate(batch):
            try:
                payload: gui_data.GuiRunResults | str = self._evaluate(
                    point, mode
                )
            except Exception as err:
                payload = f"{type(err).__name__}: {err}"
            payloads.append(payload)
            if on_result is not None:
                on_result(index, payload)
        return payloads

    def close(self) -> None:
        """Mark the session closed, as closing a real session does."""
        self.closed = True


def patch_pool_sessions(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    evaluate: Callable[[np.ndarray, str], gui_data.GuiRunResults],
    sessions: list[FakeRunSession],
    gate_path: Path | None = None,
) -> None:
    """
    Replace `module`'s ``ModelRunSession`` name with a factory of
    `FakeRunSession` s.

    The factory mirrors ``ModelRunSession``'s constructor
    parameters so the patched call sites construct it exactly as
    they would the real class; each constructed session scores its
    points through `evaluate`, gated on `gate_path` when one is
    given, and is appended to `sessions`.

    Args:
        monkeypatch: The pytest fixture doing the patching.
        module: The module whose ``ModelRunSession`` name to replace.
        evaluate: The (point, mode) evaluator each session maps.
        sessions: Collects every constructed session for inspection.
        gate_path: A file each batch waits for, or None to run freely.

    Raises:
        AttributeError:
            Raised when `module` has no ``ModelRunSession`` name to
            patch (the module does not construct sessions).
    """

    def make_session(
        csv_path: str,
        model_key: str = "",
        network: object = None,
        column_map: dict[str, str | None] | None = None,
        workers: int | None = None,
        stall_timeout: float = 0.0,
    ) -> FakeRunSession:
        session: FakeRunSession = FakeRunSession(
            csv_path, evaluate=evaluate, gate_path=gate_path
        )
        sessions.append(session)
        return session

    monkeypatch.setattr(module, "ModelRunSession", make_session)


def gradient_window(
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
    `arr_to_hydro_results` on the `window` module, which covers the
    in-process runs: the startup run and any rerun.  A job's pooled
    points are the tests' own patch: each test swaps in the session
    class the job constructs (see `patch_pool_sessions`).
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gui_window, "arr_to_results", full_fake)
    monkeypatch.setattr(gui_window, "arr_to_hydro_results", hydro_fake)
    return gui_window.MainWindow(input_data)


def only_csv(tmp_path: Path) -> Path:
    """The single ``gradients_*.csv`` in `tmp_path`'s results directory."""
    csvs: list[Path] = list((tmp_path / "results").glob("gradients_*.csv"))
    assert len(csvs) == 1
    return csvs[0]


# ------------------------------------------------------------------ #
# _SpinnerLabel (standalone)
# ------------------------------------------------------------------ #


class TestSpinnerLabel:
    """The toolbar's running-job indicator."""

    def test_spinner_label_cycles_frames(self, qapp: QApplication) -> None:
        """
        The indicator is hidden at rest and keeps a fixed 24 px
        footprint; while spinning it shows and cycles its frames on
        each timer tick, and stopping hides it again and clears the
        frame.
        """
        label = gui_window.SpinnerLabel()
        try:
            # Hidden at rest, with the fixed square footprint.
            assert label.isHidden()
            assert label.text() == ""
            assert label.minimumWidth() == gui_window.SPINNER_SIZE
            assert label.maximumWidth() == gui_window.SPINNER_SIZE
            assert label.minimumHeight() == gui_window.SPINNER_SIZE
            assert label.maximumHeight() == gui_window.SPINNER_SIZE

            # Spinning shows the current frame and advances it on
            # each tick; the ~200 ms pump below spans at least two
            # 100 ms periods, but the exact tick count depends on
            # timer drift, so only "changed" is asserted.
            label.set_spinning(True)
            assert not label.isHidden()
            first_frame: str = label.text()
            assert first_frame in gui_window.SPINNER_FRAMES
            pump(gui_window.SPINNER_INTERVAL_MS + 50)
            assert label.text() != first_frame
            assert label.text() in gui_window.SPINNER_FRAMES

            # Stopping hides the indicator and clears the frame.
            label.set_spinning(False)
            assert label.isHidden()
            assert label.text() == ""
        finally:
            label.deleteLater()
            pump()


class TestGradients:
    """The toolbar's *Gradients* action: the finite-difference jobs."""

    def test_full_mode_job(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = tmp_path / "job_gate"
        pool_full, pool_hydro = gradient_fake_factory(
            None, input_data, free[0], free[1]
        )
        win = gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert len(calls) == 1

            patch_pool_sessions(
                monkeypatch,
                model_pool,
                mode_evaluate(pool_full, pool_hydro, input_data),
                [],
                gate_path=gate,
            )
            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window.GradientWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            pump()
            # The job's points ran in the fake session (which
            # scores them without recording): only the in-process
            # startup run is recorded, and the last step completes
            # the bar.
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) <= total <= 2 * len(free)

            loaded: pd.DataFrame = pd.read_csv(
                only_csv(tmp_path), index_col="objective"
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
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
            assert win._gradients_action.isEnabled()
        finally:
            close_window(win)

    def test_hydro_mode_job(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = tmp_path / "job_gate"
        pool_full, pool_hydro = gradient_fake_factory(
            None, input_data, free[0], free[1]
        )
        win = gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._mode_action.setChecked(True)
            wait_until(
                lambda: win._worker is None
                and win._last_run_mode == MODE_HYDRO
                and win._results_are_up_to_date()
            )
            pump()
            assert len(calls) == 2

            patch_pool_sessions(
                monkeypatch,
                model_pool,
                mode_evaluate(pool_full, pool_hydro, input_data),
                [],
                gate_path=gate,
            )
            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window.GradientWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            pump()
            # The job's points ran in the fake session (which
            # scores them without recording): the two in-process
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
                only_csv(tmp_path), index_col="objective"
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
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
            assert win._gradients_action.isEnabled()
        finally:
            close_window(win)

    def test_gradient_job_failed_point_becomes_nan(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A job point that fails does not fail the job: its column
        of the difference matrix is all NaN (the missing value),
        and the completion status counts the failure.
        """
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        plain_fake = fake_results_factory(calls, input_data)
        pool_fake = fake_results_factory(None, input_data)
        failing_index: int = list(input_data.bounds).index(free[0])

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", plain_fake)
        win = gui_window.MainWindow(input_data)
        gate = tmp_path / "job_gate"
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()

            # Only the free[0] "+" perturbation exceeds the base
            # value: the base point is handed in (the results are
            # up to date), free[0] "-" and every other job point
            # equal the base there, so they run clean.
            base_value: float = win._param_panel.param_values()[free[0]]

            def pool_evaluate(
                x: np.ndarray, mode: str
            ) -> gui_data.GuiRunResults:
                if float(x[failing_index]) > base_value:
                    raise ValueError("point failed")
                return pool_fake(x, input_data)

            patch_pool_sessions(
                monkeypatch,
                model_pool,
                pool_evaluate,
                [],
                gate_path=gate,
            )
            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window.GradientWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            pump()
            # The job's points ran in the fake session (which
            # scores them without recording): only the in-process
            # startup run is recorded, and the last step completes
            # the bar.
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) <= total <= 2 * len(free)

            assert win._gradient_state is not None
            state: gui_window.GradientSnapshot = win._gradient_state
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
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
            assert win._gradients_action.isEnabled()
            only_csv(tmp_path)
        finally:
            gate.touch()
            close_window(win)

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
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        plain_fake = fake_results_factory(calls, input_data)
        pool_fake = fake_results_factory(None, input_data)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", plain_fake)
        win = gui_window.MainWindow(input_data)
        gate = tmp_path / "job_gate"
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert len(calls) == 1

            # Move free[0] to its upper bound, with the auto run
            # off so the edit does not start a model run: the
            # results go stale, so the job must run the base point
            # itself.  The base's "+" perturbation clamps at the
            # bound and coincides with the base, so it is
            # deduplicated out of the pool points and only the
            # base fails.
            win._auto_run_action.setChecked(False)
            row = win._param_panel._rows[free[0]]
            assert isinstance(row, gui_param_panel.FreeCell)
            row._spin.setValue(win._param_panel.current_bounds()[free[0]][1])
            assert not win._results_are_up_to_date()
            base_x: np.ndarray = win._current_x()

            def pool_evaluate(
                x: np.ndarray, mode: str
            ) -> gui_data.GuiRunResults:
                if np.array_equal(x, base_x):
                    raise ValueError("point failed")
                return pool_fake(x, input_data)

            patch_pool_sessions(
                monkeypatch,
                model_pool,
                pool_evaluate,
                [],
                gate_path=gate,
            )
            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window.GradientWorker)
            gate.touch()

            wait_until(
                lambda: win._worker is None
                and win._status_label.text().startswith("Gradients failed")
            )
            pump()
            assert (
                win._status_label.text()
                == "Gradients failed: ValueError: point failed"
            )
            assert win._gradient_state is None
            assert win._figure_panel.page_count() == 8
            # The base never arrived, so the results stay stale:
            # the save button stays down and the run button stays
            # up.
            assert not win._save_action.isEnabled()
            assert win._run_action.isEnabled()
            assert win._gradients_action.isEnabled()
            # The job's points ran in the fake session (which
            # scores them without recording): only the in-process
            # startup run is recorded, and no matrix was written.
            assert len(calls) == 1
            assert list((tmp_path / "results").glob("gradients_*.csv")) == []
        finally:
            gate.touch()
            close_window(win)

    def test_no_free_parameters(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()

            def no_free() -> dict[str, tuple[float, float]]:
                return {
                    key: (value, value)
                    for key, value in win._param_panel.param_values().items()
                }

            monkeypatch.setattr(win._param_panel, "current_bounds", no_free)
            win._on_gradients_clicked()
            pump()
            assert (
                win._status_label.text()
                == "No free parameters to differentiate."
            )
            assert win._worker is None
            assert len(calls) == 1
            assert win._gradients_action.isEnabled()
        finally:
            close_window(win)

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
        toolbar's spinner spins while the job runs; both go back to
        their rest state when the job finishes.
        """
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        win = gradient_window(
            input_data, tmp_path, monkeypatch, full_fake, hydro_fake
        )
        gate = tmp_path / "job_gate"
        try:
            # Let the window's startup run finish first, so the
            # job's base point is handed in rather than rerun.
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert win._results_are_up_to_date()

            # The job's session batches block on a file gate until
            # the test has connected to the worker's progress_step
            # signal: the worker is created by the click, so it
            # cannot be connected in advance, and the gate keeps
            # the session from completing (and emitting a step
            # for) a point before the connection.
            pool_full, pool_hydro = gradient_fake_factory(
                None, input_data, free[0], free[1]
            )
            patch_pool_sessions(
                monkeypatch,
                model_pool,
                mode_evaluate(pool_full, pool_hydro, input_data),
                [],
                gate_path=gate,
            )

            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window.GradientWorker)

            steps: list[tuple[int, int]] = []
            spinner_up: list[bool] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))
                spinner_up.append(not win._job_spinner.isHidden())

            worker.progress_step.connect(on_step)
            gate.touch()

            wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            pump()

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
            close_window(win)


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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()

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
            pump()
            assert win._objective_terms == frozenset({"Streamflow: KGE"})
            assert win._objective_model.is_selected("Streamflow: KGE")

            qapp.sendEvent(viewport, event)
            pump()
            assert win._objective_terms == frozenset()
            assert not win._objective_model.is_selected("Streamflow: KGE")

            win._objective_terms = None
            win._on_objective_toggled("DOC: R\u00b2")
            assert win._objective_terms is None
        finally:
            close_window(win)

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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            win._auto_run_action.setChecked(False)

            win._on_objective_toggled("Streamflow: KGE")
            win._on_objective_toggled("DOC: KGE")
            assert win._objective_terms == frozenset(
                {"Streamflow: KGE", "DOC: KGE"}
            )

            win._mode_action.setChecked(True)
            assert win._objective_terms == frozenset({"Streamflow: KGE"})
            assert win._objective_model.is_selected("Streamflow: KGE")
            assert not win._objective_model.is_selected("DOC: KGE")

            win._mode_action.setChecked(False)
            assert win._objective_terms == frozenset({"Streamflow: KGE"})
            assert not win._objective_model.is_selected("DOC: KGE")
        finally:
            close_window(win)


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
# The toolbar's *Line search* action: the steepest-ascent jobs
# ------------------------------------------------------------------ #


class TestLineSearch:
    """The toolbar's *Line search* action: the steepest-ascent jobs."""

    def test_full_mode_job(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A search with no gradient job computes its own direction."""
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = tmp_path / "job_gate"
        pool_full, pool_hydro = gradient_fake_factory(
            None, input_data, free[0], free[1]
        )
        win = gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert len(calls) == 1

            # The fake shifts the streamflow objectives by 2 * x of
            # free[0] alone, so the steepest ascent of Streamflow's
            # KGE is the unit direction along free[0]'s axis and the
            # best point is free[0]'s upper bound.
            mid: float = win._param_panel.param_values()[free[0]]
            high: float = win._param_panel.current_bounds()[free[0]][1]

            patch_pool_sessions(
                monkeypatch,
                gui_window,
                mode_evaluate(pool_full, pool_hydro, input_data),
                [],
                gate_path=gate,
            )
            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            worker = win._worker
            assert isinstance(worker, gui_window.LineSearchWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            pump()
            # The base point's results were handed in, the
            # direction came from forward differences over the
            # base point's own perturbations (one session batch),
            # and the wave schedule then scored the line in
            # further batches: all the job's points ran in the
            # fake session, so only the in-process startup run is
            # recorded, and the last step completes the bar.
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) + 8 <= total <= len(free) + 16

            assert win._line_search_state is not None
            profile: pd.DataFrame = win._line_search_state.profile
            assert 8 <= len(profile) <= 16
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
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
            assert win._gradients_action.isEnabled()
            assert win._line_search_action.isEnabled()

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
            gate.touch()
            close_window(win)

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
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = tmp_path / "job_gate"
        pool_full, pool_hydro = gradient_fake_factory(
            None, input_data, free[0], free[1]
        )
        win = gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert len(calls) == 1

            # The fake shifts the streamflow objectives by 2 * x of
            # free[0] alone, so the steepest ascent of Streamflow's
            # KGE is the unit direction along free[0]'s axis and
            # the best point is free[0]'s upper bound.
            mid: float = win._param_panel.param_values()[free[0]]
            high: float = win._param_panel.current_bounds()[free[0]][1]

            # The feasible interval from the midpoint base is
            # symmetric, so the wave-1 grid of the engine schedule
            # is predictable even though the direction carries
            # forward-difference noise: flag the sixth grid step's
            # parameter value (tolerantly, since the fake objective
            # depends only on this coordinate) as the point that
            # fails, and delegate every other point to the startup
            # fakes' non-recording twins.
            low: float = win._param_panel.current_bounds()[free[0]][0]
            keys: list[str] = list(win._param_panel.current_bounds())
            failing_position: int = keys.index(free[0])
            wave_grid: np.ndarray = search_t_points(low - mid, high - mid, 7)
            failing_value: float = mid + float(wave_grid[5])

            def pool_evaluate(
                x: np.ndarray, mode: str
            ) -> gui_data.GuiRunResults:
                if abs(float(x[failing_position]) - failing_value) <= (
                    1e-9 * (high - low)
                ):
                    raise ValueError("point failed")
                if mode == MODE_HYDRO:
                    return pool_hydro(x, input_data)
                return pool_full(x, input_data)

            patch_pool_sessions(
                monkeypatch,
                gui_window,
                pool_evaluate,
                [],
                gate_path=gate,
            )
            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            worker = win._worker
            assert isinstance(worker, gui_window.LineSearchWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            pump()
            # The base point's results were handed in, the
            # direction came from forward differences over the
            # base point's own perturbations (one session batch),
            # and the wave schedule then scored the line in
            # further batches: all the job's points ran in the
            # fake session, so only the in-process startup run is
            # recorded, and the last step completes the bar.
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) + 8 <= total <= len(free) + 16

            assert win._line_search_state is not None
            state: gui_window.LineSearchSnapshot = win._line_search_state
            profile: pd.DataFrame = state.profile
            assert 8 <= len(profile) <= 16
            values: np.ndarray = profile["value"].to_numpy(dtype=np.float64)
            # The failed grid point is the profile's only NaN row,
            # at its grid step; every other row is finite.
            nan_rows: np.ndarray = np.flatnonzero(np.isnan(values))
            assert len(nan_rows) == 1
            assert float(profile.index[nan_rows[0]]) == pytest.approx(
                float(wave_grid[5])
            )
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
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()

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
            close_window(win)

    def test_worker_reuses_the_base_results_at_the_zero_step(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        The wave grid's point at ``t = 0`` is the base point itself:
        the worker reuses the handed-in base results for that
        profile row instead of scoring the point through the
        session, and every other grid point is pooled exactly once.
        """
        monkeypatch.chdir(tmp_path)
        bounds: dict[str, tuple[float, float]] = dict(input_data.bounds)
        base_x: np.ndarray = midpoint_vector(input_data)
        steps: list[Perturbation] = free_parameter_steps(base_x, bounds)
        free = free_keys(input_data)
        keys: list[str] = list(bounds)
        direction: np.ndarray = np.zeros(len(keys))
        direction[keys.index(free[0])] = 0.6
        direction[keys.index(free[1])] = 0.8
        hydro, rt = synthetic_results(input_data)
        base_results: gui_data.GuiRunResults = gui_data.GuiRunResults(
            hydro, rt
        )

        def flat_evaluate(
            _x: np.ndarray, _mode: str
        ) -> gui_data.GuiRunResults:
            point_hydro, point_rt = synthetic_results(input_data)
            point_hydro.objective_functions = (
                point_hydro.objective_functions + 0.1
            )
            return gui_data.GuiRunResults(point_hydro, point_rt)

        sessions: list[FakeRunSession] = []
        patch_pool_sessions(monkeypatch, gui_window, flat_evaluate, sessions)
        worker: gui_window.LineSearchWorker = gui_window.LineSearchWorker(
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
            assert 8 <= len(profile) <= 16
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
            # One session served the whole search: it saw exactly
            # the grid points other than the base, once each, and
            # closed when the search finished.
            assert len(sessions) == 1
            pooled: int = sum(len(batch) for batch in sessions[0].batches)
            assert pooled == len(profile) - 1
            for batch in sessions[0].batches:
                for point in batch:
                    assert not np.array_equal(point, base_x)
            assert sessions[0].closed
        finally:
            worker.deleteLater()
            pump()

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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()

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
            alt_hydro, alt_rt = synthetic_results(input_data)
            outcome = gui_window.LineSearchOutcome(
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
            state: gui_window.LineSearchSnapshot = win._line_search_state
            assert state.profile is profile
            assert state.csv_path == csv_path
            assert win._figure_panel.page_count() == 9
            assert win._figure_panel._titles[-1] == "Line search (full)"

            assert win._line_search_action.isEnabled()
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
            assert win._status_label.text() == (
                "Line search finished in 3.0 s: no improvement found "
                "(the best value equals the base value); the "
                "parameters are unchanged and the profile was saved "
                f"to {csv_path}."
            )
        finally:
            close_window(win)

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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()

            before_values: dict[str, float] = (
                win._param_panel.param_values()
            )
            base_x: np.ndarray = win._current_x()
            free = free_keys(input_data)
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
            alt_hydro, alt_rt = synthetic_results(input_data)
            alt_results: gui_data.GuiRunResults = gui_data.GuiRunResults(
                alt_hydro, alt_rt
            )
            outcome = gui_window.LineSearchOutcome(
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
            state: gui_window.LineSearchSnapshot = win._line_search_state
            assert state.profile is profile
            assert state.csv_path == csv_path
            assert win._figure_panel.page_count() == 9
            assert win._figure_panel._titles[-1] == "Line search (full)"

            assert win._line_search_action.isEnabled()
            assert win._save_action.isEnabled()
            assert not win._run_action.isEnabled()
            assert win._status_label.text() == (
                f"Line search finished in 3.0 s: best t = 0.5000 "
                f"(Streamflow: KGE = 0.905); the best point's "
                f"parameters are in the panel and the profile was "
                f"saved to {csv_path}."
            )
        finally:
            close_window(win)

    def test_line_search_keeps_the_figure_page(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A finished search adds its profile page without moving the
        user off the figure they were on: the tab raised before the
        job is still the raised tab after the figure rebuild, both
        in the panel's tracker and on the visible bar.  Qt settles
        the rebuilt tabify chain on a later event-loop pass and
        raises the first dock there; the rebuild's deferred page
        re-assertion is what survives it.
        """
        from ..gui.line_search import write_line_search_csv

        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            win.show()
            pump()
            panel: gui_figure_panel.FigurePanel = win._figure_panel
            assert panel.page_count() == 8

            # The user is on a middle page: a click on its tab.
            page: int = 5
            figure_tab_bar(win).setCurrentIndex(page)
            pump(10)
            title: str = panel._titles[page]
            assert panel.current_page() == page
            assert visible_tab(win) == (page, title)

            # The same rising, adopted profile as in
            # `test_improvement_is_still_adopted`.
            base_x: np.ndarray = win._current_x()
            keys: list[str] = list(win._param_panel.current_bounds())
            free: list[str] = free_keys(input_data)
            x_best: np.ndarray = np.array(base_x)
            x_best[keys.index(free[0])] += 0.001
            t_points: np.ndarray = np.linspace(-0.5, 0.5, 25)
            t_points[int(np.argmin(np.abs(t_points)))] = 0.0
            values: np.ndarray = 0.9 + 0.01 * t_points
            profile: pd.DataFrame = pd.DataFrame(
                {"value": values, "Streamflow: KGE": values},
                index=t_points,
            )
            csv_path: Path = write_line_search_csv(profile)
            alt_hydro, alt_rt = synthetic_results(input_data)
            outcome = gui_window.LineSearchOutcome(
                error=None,
                profile=profile,
                x_best=x_best,
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
            pump()

            # The profile page was appended, the outcome was adopted...
            assert panel.page_count() == 9
            assert "parameters are in the panel" in win._status_label.text()
            # ...and the user's page survived the rebuild at its title.
            assert panel._titles[page] == title
            assert panel.current_page() == page
            assert visible_tab(win) == (page, title)
        finally:
            close_window(win)

    def test_line_search_adoption_keeps_the_param_view(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Adopting a better point updates the panel in place: the
        scroll area stays scrolled where the user had it (the moved
        reactive-transport cell stays in the view), every cell
        widget is the same object as before (no group rebuild), and
        the moved cell's spin box shows the adopted value.
        """
        from ..gui.line_search import write_line_search_csv

        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            win.show()
            win.resize(1100, 800)
            pump()

            panel: gui_param_panel.ParamPanel = win._param_panel
            scrollbar: QScrollBar = panel.verticalScrollBar()
            assert scrollbar.maximum() > 0
            # The user is looking at the bottom of the panel, where
            # the reactive-transport groups sit.
            scrollbar.setValue(scrollbar.maximum())
            assert scrollbar.value() == scrollbar.maximum()

            # The first free reactive-transport parameter: one of
            # those groups' cells.  The search moves it two spin
            # steps.
            rt_fields: frozenset[str] = frozenset(
                RT_DIM_PARAMS + MINERAL_PARAMS
            )
            rt_key: str = next(
                key
                for key in free_keys(input_data)
                if key.partition(".")[2].split("[", 1)[0] in rt_fields
            )
            rt_bounds: tuple[float, float] = panel.current_bounds()[rt_key]
            lo, hi = rt_bounds
            step: float = 10.0 ** (-bound_decimals(lo, hi))
            before_cells: dict[str, object] = {
                key: cell for key, cell in panel._rows.items()
            }
            before_values: dict[str, float] = panel.param_values()
            base_x: np.ndarray = win._current_x()
            keys: list[str] = list(panel.current_bounds())
            x_best: np.ndarray = np.array(base_x)
            x_best[keys.index(rt_key)] += 2.0 * step

            # A rising profile, as in `test_improvement_is_still_adopted`.
            t_points: np.ndarray = np.linspace(-0.5, 0.5, 25)
            t_points[int(np.argmin(np.abs(t_points)))] = 0.0
            values: np.ndarray = 0.9 + 0.01 * t_points
            profile: pd.DataFrame = pd.DataFrame(
                {"value": values, "Streamflow: KGE": values},
                index=t_points,
            )
            csv_path: Path = write_line_search_csv(profile)
            alt_hydro, alt_rt = synthetic_results(input_data)
            outcome = gui_window.LineSearchOutcome(
                error=None,
                profile=profile,
                x_best=x_best,
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
            pump()

            # The best point's values are adopted...
            assert "parameters are in the panel" in win._status_label.text()
            after_values: dict[str, float] = panel.param_values()
            expected: float = min(
                max(before_values[rt_key] + 2.0 * step, lo), hi
            )
            assert after_values[rt_key] == pytest.approx(expected)
            for key in before_values:
                if key != rt_key:
                    assert after_values[key] == before_values[key]

            # ...on the same cell widgets, so no group rebuild
            # happened...
            for key, cell in before_cells.items():
                assert panel._rows[key] is cell
            # ...the scroll area kept its position...
            assert scrollbar.value() > 0
            # ...and the moved cell's spin box shows the adopted value.
            spin: QDoubleSpinBox | None = panel._rows[rt_key].findChild(
                QDoubleSpinBox
            )
            assert spin is not None
            assert spin.value() == pytest.approx(after_values[rt_key], abs=step)
        finally:
            close_window(win)

    def test_reuses_gradient_snapshot(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A finished gradient job's row supplies the direction for free."""
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gradient_gate = tmp_path / "gradient_gate"
        pool_full, pool_hydro = gradient_fake_factory(
            None, input_data, free[0], free[1]
        )
        win = gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert len(calls) == 1

            patch_pool_sessions(
                monkeypatch,
                model_pool,
                mode_evaluate(pool_full, pool_hydro, input_data),
                [],
                gate_path=gradient_gate,
            )
            win._on_gradients_clicked()
            worker = win._worker
            assert isinstance(worker, gui_window.GradientWorker)

            gradient_steps: list[tuple[int, int]] = []

            def on_gradient_step(done: int, total: int) -> None:
                gradient_steps.append((done, total))

            worker.progress_step.connect(on_gradient_step)
            gradient_gate.touch()
            wait_until(
                lambda: win._gradient_state is not None
                and win._worker is None
            )
            pump()
            # The job's points ran in the fake session (which
            # scores them without recording): only the in-process
            # startup run is recorded, and the last step completes
            # the bar.
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

            # The search runs in a session of its own, gated on its
            # own file so the test can connect to the worker's
            # progress before any point completes.
            search_gate = tmp_path / "search_gate"
            patch_pool_sessions(
                monkeypatch,
                gui_window,
                mode_evaluate(pool_full, pool_hydro, input_data),
                [],
                gate_path=search_gate,
            )

            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            search_worker = win._worker
            assert isinstance(search_worker, gui_window.LineSearchWorker)

            search_steps: list[tuple[int, int]] = []

            def on_search_step(done: int, total: int) -> None:
                search_steps.append((done, total))

            search_worker.progress_step.connect(on_search_step)
            search_gate.touch()
            wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            pump()
            # No direction differences and no base re-run: the
            # snapshot's row supplies the direction and the base
            # point's results are handed in, so the progress total
            # is exactly the wave schedule's 15-point budget — and
            # none of the points run in-process, so the recorded
            # calls stay put.
            assert len(calls) == 1
            assert search_steps
            assert all(
                a[0] < b[0] for a, b in zip(search_steps, search_steps[1:])
            )
            assert len({total for _done, total in search_steps}) == 1
            search_total: int = search_steps[0][1]
            assert search_steps[-1] == (search_total, search_total)
            assert search_total == 15

            assert win._line_search_state is not None
            profile: pd.DataFrame = win._line_search_state.profile
            assert 8 <= len(profile) <= 16
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
            close_window(win)

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
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = tmp_path / "search_gate"
        pool_full, pool_hydro = gradient_fake_factory(
            None, input_data, free[0], free[1]
        )
        win = gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert len(calls) == 1

            terms: frozenset[str] = frozenset(
                {"Streamflow: KGE", "Streamflow: NSE"}
            )
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            patch_pool_sessions(
                monkeypatch,
                gui_window,
                mode_evaluate(pool_full, pool_hydro, input_data),
                [],
                gate_path=gate,
            )
            win._start_line_search(terms)
            worker = win._worker
            assert isinstance(worker, gui_window.LineSearchWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            pump()
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            assert len(free) + 8 <= total <= len(free) + 16

            state = win._line_search_state
            assert state is not None
            assert state.mode == MODE_FULL
            assert state.terms == ["Streamflow: KGE", "Streamflow: NSE"]
            profile: pd.DataFrame = state.profile
            assert 8 <= len(profile) <= 16
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
            lines = line_artists(figure.axes[0])
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
            close_window(win)

    def test_line_search_in_hydro_mode(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        The same checkbox selection searches in hydro mode: the
        session runs the hydro objectives and the page and the
        status report the hydro search.
        """
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, _ = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        win = make_window(input_data, tmp_path, monkeypatch, full_fake)
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert len(calls) == 1

            # No auto run after the mode switch: the search's
            # session batches are the only pooled work, held on
            # the file gate until the test releases them.  The
            # session's evaluator touches a marker file per point
            # so the test can see the points ran through it.
            win._auto_run_action.setChecked(False)
            win._mode_action.setChecked(True)
            file_gate = tmp_path / "job_gate"
            pool_full, pool_hydro = gradient_fake_factory(
                None, input_data, free[0], free[1]
            )

            def marking_evaluate(
                x: np.ndarray, mode: str
            ) -> gui_data.GuiRunResults:
                (tmp_path / f"hydro_{float(x.sum()):.6f}").touch()
                if mode == MODE_HYDRO:
                    return pool_hydro(x, input_data)
                return pool_full(x, input_data)

            patch_pool_sessions(
                monkeypatch,
                gui_window,
                marking_evaluate,
                [],
                gate_path=file_gate,
            )

            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            assert isinstance(
                win._worker, gui_window.LineSearchWorker
            )

            file_gate.touch()
            wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            pump()
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
            close_window(win)

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
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, hydro_fake = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = tmp_path / "search_gate"
        pool_full, pool_hydro = gradient_fake_factory(
            None, input_data, free[0], free[1]
        )
        win = gradient_window(
            input_data,
            tmp_path,
            monkeypatch,
            full_fake,
            hydro_fake,
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert len(calls) == 1

            # A snapshot that lacks the selected term, with a csv
            # file that exists so only the term lookup can fail.
            results_dir: Path = tmp_path / "results"
            results_dir.mkdir(exist_ok=True)
            csv_path: Path = results_dir / "gradients_fake.csv"
            csv_path.touch()
            win._gradient_state = gui_window.GradientSnapshot(
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
            patch_pool_sessions(
                monkeypatch,
                gui_window,
                mode_evaluate(pool_full, pool_hydro, input_data),
                [],
                gate_path=gate,
            )
            win._start_line_search(terms)
            worker = win._worker
            assert isinstance(worker, gui_window.LineSearchWorker)

            steps: list[tuple[int, int]] = []

            def on_step(done: int, total: int) -> None:
                steps.append((done, total))

            worker.progress_step.connect(on_step)
            gate.touch()

            wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            pump()
            assert len(calls) == 1
            assert steps
            assert all(a[0] < b[0] for a, b in zip(steps, steps[1:]))
            assert len({total for _done, total in steps}) == 1
            total: int = steps[0][1]
            assert steps[-1] == (total, total)
            # The base point's results were handed in and the
            # forward differences run for every free parameter
            # (nothing dedups at the midpoint base), so the total
            # is the difference batch plus the wave schedule's
            # 15-point budget.
            assert total == len(free) + 15

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
            close_window(win)

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

        win = make_window(input_data, tmp_path, monkeypatch, failing)
        try:
            wait_until(
                lambda: win._worker is None
                and win._status_label.text() == "Run failed: ValueError: boom"
            )
            pump()
            win._on_line_search_clicked()
            assert (
                win._status_label.text()
                == "No objective values to search yet: run the model first."
            )
            assert win._worker is None
            assert win._line_search_state is None
        finally:
            close_window(win)

    def test_no_free_parameters(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With every range collapsed, the selection is refused."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
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
            pump()
            assert (
                win._status_label.text() == "No free parameters to search."
            )
            assert win._worker is None
            assert win._line_search_state is None
            assert len(calls) == 1
            assert win._line_search_action.isEnabled()
        finally:
            close_window(win)

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
        free = free_keys(input_data)
        calls: list[np.ndarray] = []
        full_fake, _hydro_fake = gradient_fake_factory(
            calls, input_data, free[0], free[1]
        )
        gate = threading.Event()

        def gated_fake(
            x: np.ndarray, input_data: gui_data.GuiInputData
        ) -> gui_data.GuiRunResults:
            # The gate blocks the model calls until the test
            # releases it, so the run stays in flight while the
            # refusal is asserted, and then finishes on demand.
            gate.wait(timeout=60)
            return full_fake(x, input_data)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", gated_fake)
        monkeypatch.setattr(gui_window, "arr_to_hydro_results", gated_fake)
        # The search job's points run in the session it builds:
        # the fake session scores them in-process, so the test's
        # second half does not fork real workers.
        pool_full, _ = gradient_fake_factory(
            None, input_data, free[0], free[1]
        )
        patch_pool_sessions(
            monkeypatch,
            gui_window,
            mode_evaluate(pool_full, pool_full, input_data),
            [],
        )
        win = gui_window.MainWindow(input_data)
        try:
            # The window's startup run blocks on the gate: a model
            # run in flight, the only job the window knows.
            wait_until(
                lambda: win._worker is not None
                and win._worker.isRunning()
            )
            assert isinstance(win._worker, gui_window.RunWorker)
            run_id: int = id(win._worker)

            # The direct path is guarded against the in-flight run:
            # refused with a status message, the run's worker
            # reference untouched, nothing adopted or started.
            terms: frozenset[str] = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            pump()
            assert win._worker is not None
            assert id(win._worker) == run_id
            assert win._worker.isRunning()
            assert not isinstance(win._worker, gui_window.LineSearchWorker)
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
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            assert win._results_are_up_to_date()
            assert win._line_search_state is None
            assert win._figure_panel.page_count() == 8
            assert re.fullmatch(
                r"Run 1 finished in \d+\.\d s\.", win._status_label.text()
            ) is not None
            assert len(calls) == 1
            assert np.array_equal(calls[0], midpoint_vector(input_data))

            # With no job in flight, the same call starts the
            # search: the base point's results are handed in, and
            # the worker computes the direction and waves the
            # search through its session.
            terms = frozenset({"Streamflow: KGE"})
            win._objective_terms = terms
            win._objective_model.set_selected(terms)
            win._start_line_search(terms)
            assert isinstance(win._worker, gui_window.LineSearchWorker)
            wait_until(
                lambda: win._line_search_state is not None
                and win._worker is None
            )
            pump()
            # The job's points ran in the fake session (which
            # scores them without recording): only the in-process
            # startup run is recorded.
            assert len(calls) == 1
        finally:
            gate.set()
            close_window(win)


class TestCalibration:
    """The toolbar's *Calibrate* action and its differential-evolution job."""

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
        hydro, _rt = synthetic_results(input_data)
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
        x: np.ndarray = midpoint_vector(input_data)
        free_name: str = all_names[0]
        free_high: float = float(x[0]) + 0.1
        bounds: dict[str, tuple[float, float]] = {
            name: (float(x[index]), float(x[index]))
            for index, name in enumerate(all_names)
        }
        bounds[free_name] = (float(x[0]) - 0.1, free_high)

        hydro, _rt = synthetic_results(input_data)
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
        worker: gui_calibration.CalibrationWorker = (
            gui_calibration.CalibrationWorker(
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
        full_fake = fake_results_factory(calls, input_data)
        hydro_fake = fake_results_factory(calls, input_data, hydro_only=True)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui_window, "arr_to_results", full_fake)
        monkeypatch.setattr(
            gui_window, "arr_to_hydro_results", hydro_fake
        )
        win = gui_window.MainWindow(input_data)
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            pump()
            # Full mode: the action is disabled and the direct call
            # is refused with a message.
            assert not win._calibrate_action.isEnabled()
            win._on_calibrate_clicked()
            assert win._status_label.text() == (
                "Calibration optimizes the hydrology model only; "
                "switch the toolbar's *Hydrology only* toggle on."
            )
            assert win._worker is None

            # Hydrology-only mode: the action comes up after its run.
            win._mode_action.setChecked(True)
            wait_until(
                lambda: win._current_mode() == MODE_HYDRO
                and win._worker is None
                and win._results is not None
                and win._results.rt is None
            )
            pump()
            assert win._calibrate_action.isEnabled()

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
            hydro, _rt = synthetic_results(input_data)

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
            assert isinstance(win._worker, gui_calibration.CalibrationWorker)
            wait_until(
                lambda: win._worker is None
                and "Calibration finished" in win._status_label.text()
            )
            pump()

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
            assert win._calibrate_action.isEnabled()
        finally:
            close_window(win)


class TestModelSelection:
    """The toolbar's model selector and the state a switch resets."""

    def test_starts_on_lateral_and_titles_it(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The window opens on the lateral model and names it."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            assert win._model_combo.currentData() == "lateral_three_layer"
            assert win.windowTitle() == (
                "East River calibration — Lateral three-layer"
            )
        finally:
            close_window(win)

    def test_model_combo_lists_the_four_models(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The selector offers exactly the four curated models, in order."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
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
            close_window(win)

    def test_switching_model_resets_panel_and_clears_results(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A switch reloads the panel, clears the figures, and re-titles."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            win._model_combo.setCurrentIndex(
                win._model_combo.findData("hbv")
            )
            pump()

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
            close_window(win)

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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._model_combo.setCurrentIndex(
                win._model_combo.findData("three_layer")
            )
            wait_until(
                lambda: win._input_data.spec.key == "three_layer"
                and win._worker is None
                and win._results is not None
            )
            pump(50)

            assert win._figure_panel.page_count() == 8
            assert win._figure_panel._titles[6] == "Zone concentrations"
        finally:
            close_window(win)

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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            hbv_input: gui_data.GuiInputData = model_inputs["hbv"]
            write_saved_run(tmp_path, input_data)
            write_saved_run(
                tmp_path,
                hbv_input,
                state=full_state(hbv_input, model_key="hbv"),
            )

            win._auto_run_action.setChecked(False)
            win._model_combo.setCurrentIndex(
                win._model_combo.findData("hbv")
            )
            pump()

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
            close_window(win)

    def test_gradient_outcome_from_another_model_is_discarded(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A gradient outcome from another model never reaches the panel."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            outcome = gui_window.GradientOutcome(
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
            assert win._gradients_action.isEnabled()
            assert win._line_search_action.isEnabled()
        finally:
            close_window(win)

    def test_saves_the_selected_model(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Saving under HBV writes an HBV model and records the key."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            win._model_combo.setCurrentIndex(
                win._model_combo.findData("hbv")
            )
            pump()
            # Run once so there are results to save for the new model.
            win._start_run(win._current_x())
            wait_until(
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
            close_window(win)


# Pinned W1 finding line (Task 2's implementation notes).
W1_LINE: str = (
    "W1: no H+ primary species: the speciation solve has "
    "no charge-balance row"
)


class TestNetworkEditorDialog:
    """
    The T9 modal editor dialog: seeding and preset detection, the
    validation bar, the checkbox-driven remove/re-add flow, Apply
    and the resulting fingerprint, and the preset re-seed
    confirmation (the window's Yes/No idiom, faked with
    `FakeMessageBox`).
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
            pump()

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
            assert dialog._warning_label.text() == W1_LINE
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
            pump()

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
            pump()

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
            gui_network_editor, "QMessageBox", FakeMessageBox
        )
        try:
            FakeMessageBox.answer = FakeMessageBox.StandardButton.Yes
            dialog._dirty = True
            dialog._preset_combo.setCurrentIndex(
                dialog._preset_combo.findData("carbon")
            )
            assert not dialog._dirty
            assert dialog._preset_combo.currentData() == "carbon"
            assert dialog._ok_label.text() == (
                "OK — 8 species, 1 minerals, 1 exchange"
            )
            FakeMessageBox.answer = FakeMessageBox.StandardButton.No
            dialog._dirty = True
            dialog._preset_combo.setCurrentIndex(
                dialog._preset_combo.findData("chloride")
            )
            assert dialog._dirty
            assert dialog._preset_combo.currentData() == "carbon"
        finally:
            FakeMessageBox.answer = FakeMessageBox.StandardButton.No
            dialog.deleteLater()
            pump()


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
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
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
            close_window(win)

    def test_switching_to_carbon_resets_the_window(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A switch keeps the model, swaps the network, and clears state."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            win._network_combo.setCurrentIndex(
                win._network_combo.findData("carbon")
            )
            pump()

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
            close_window(win)

    def test_loading_a_preset_matching_file_selects_the_preset(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A file carrying the carbon preset's network selects it by key."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)

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
            pump()

            carbon_fingerprint: str = gui_data.fingerprint_network(
                get_carbon_network()
            )
            assert gui_data.fingerprint_network(win._network) == (
                carbon_fingerprint
            )
            assert win._network_combo.currentData() == "carbon"
            assert win._custom_network_name is None
        finally:
            close_window(win)

    def test_loading_an_invalid_file_keeps_the_network(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A file that fails to load or validate is reported and ignored."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
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
                gui_window, "QMessageBox", FakeMessageBox
            )
            FakeMessageBox.critical_calls.clear()
            win._on_load_network()
            pump()

            assert len(FakeMessageBox.critical_calls) == 1
            assert FakeMessageBox.critical_calls[0][0] == "Load network"
            assert gui_data.fingerprint_network(win._network) == before
            assert win._network_combo.currentData() == (
                network_specs.DEFAULT_NETWORK_KEY
            )
        finally:
            close_window(win)

    def test_loading_a_custom_file_adds_a_custom_entry(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-preset file becomes the selector's `custom: <stem>` entry."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)

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
                gui_network_editor.preset_fingerprint_map()
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
            pump()

            assert gui_data.fingerprint_network(win._network) == (
                fingerprint
            )
            assert win._network_combo.currentData() == fingerprint
            assert win._custom_network_name == "my_net"
            assert win._network_combo.itemText(
                win._network_combo.findData(fingerprint)
            ) == "custom: my_net"
        finally:
            close_window(win)

    def test_edit_cancel_keeps_the_network(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancelling the editor dialog leaves the window untouched."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            before: str = gui_data.fingerprint_network(win._network)

            def fake_edit(
                network: pt.ReactionNetwork,
                database: pt.ChemicalDatabase,
                parent: QWidget | None,
            ) -> pt.ReactionNetwork | None:
                return None

            monkeypatch.setattr(gui_window, "edit_network", fake_edit)
            win._on_edit_network()
            pump()

            assert gui_data.fingerprint_network(win._network) == before
            assert win._network_combo.currentData() == (
                network_specs.DEFAULT_NETWORK_KEY
            )
            assert win._custom_network_name is None
        finally:
            close_window(win)

    def test_edit_apply_adds_an_edited_entry(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Applying an edit of a preset gets the `custom: edited` entry."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)

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
            pump()

            assert gui_data.fingerprint_network(win._network) == (
                fingerprint
            )
            assert win._network_combo.currentData() == fingerprint
            assert win._custom_network_name == "edited"
            assert win._network_combo.itemText(
                win._network_combo.findData(fingerprint)
            ) == "custom: edited"
        finally:
            close_window(win)

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
        assert gui_window.run_matches_network(run_a, default_fingerprint)
        assert not gui_window.run_matches_network(run_a, carbon_fingerprint)
        assert gui_window.run_matches_network(run_b, "abc123")
        assert not gui_window.run_matches_network(run_b, default_fingerprint)
        assert not gui_window.run_matches_network(run_b, carbon_fingerprint)

    def test_switch_filters_history_by_model_and_network(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The history list keeps only runs of the same model and network."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)

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
            pump()
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
            pump()
            actual = [
                item.run_dir.name
                for item in win._history_items
                if item.run_dir is not None
            ]
            assert actual == [other_network.name, old_run.name]
        finally:
            close_window(win)

    def test_on_save_network_writes_the_file(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Save network as writes the canonical file; the window is untouched."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)

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
                gui_window, "QMessageBox", FakeMessageBox
            )
            FakeMessageBox.information_calls.clear()
            win._on_save_network()

            assert target.is_file()
            saved: pt.ReactionNetwork = pt.ReactionNetwork.from_file(
                str(target)
            )
            saved_fingerprint: str = gui_data.fingerprint_network(saved)
            assert saved_fingerprint == gui_data.fingerprint_network(
                win._network
            )
            assert FakeMessageBox.information_calls == [
                ("Save network", f"Network written to {target}.")
            ]
            assert win._network_combo.currentData() == (
                network_specs.DEFAULT_NETWORK_KEY
            )
            assert win._custom_network_name is None
        finally:
            close_window(win)

    def test_on_save_run_persists_the_network(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Saving a run writes network.yaml and the fingerprint to state."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            win._network_combo.setCurrentIndex(
                win._network_combo.findData("carbon")
            )
            pump()
            # Run once so there are results to save for the new
            # network.
            win._start_run(win._current_x())
            wait_until(
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
            close_window(win)

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
        loaded: pt.ReactionNetwork | None = gui_window.run_network(good)
        assert loaded is not None
        assert gui_data.fingerprint_network(loaded) == (
            gui_data.fingerprint_network(get_carbon_network())
        )

        missing: Path = tmp_path / "missing"
        missing.mkdir()
        assert gui_window.run_network(missing) is None

        broken: Path = tmp_path / "broken"
        broken.mkdir()
        (broken / "network.yaml").write_text(":::")
        assert gui_window.run_network(broken) is None
        assert "Could not load" in capsys.readouterr().err


# ------------------------------------------------------------------ #
# The column-mapping dialog (standalone)
# ------------------------------------------------------------------ #


class TestColumnMappingDialog:
    """The role-per-row mapping dialog of the Load data flow."""

    @staticmethod
    def _columns() -> list[str]:
        return [
            "date",
            "ppt",
            "tair",
            "evap",
            "flow",
            "doc_molL",
            "ca_molL",
            "dic_molL",
            "unmapped_column",
        ]

    @staticmethod
    def _prefill() -> dict[str, str | None]:
        return {
            "date": "date",
            "precip": "ppt",
            "temp": "tair",
            "pet": "evap",
            "q_meas": "flow",
            "DOC": "doc_molL",
            "Ca++": "ca_molL",
            "DIC": "dic_molL",
        }

    @staticmethod
    def _make(
        qapp: QApplication,
        prefill: dict[str, str | None],
    ) -> gui_data_dialog.ColumnMappingDialog:
        return gui_data_dialog.ColumnMappingDialog(
            TestColumnMappingDialog._columns(),
            get_soc_calcite_network(),
            prefill,
        )

    def test_prefill_applied(
        self, qapp: QApplication
    ) -> None:
        """Prefilled roles show their column in the row's combo, and
        unmentioned rows stay on their sentinel."""
        dialog = self._make(
            qapp, {"precip": "ppt", "DOC": "doc_molL"}
        )
        try:
            assert dialog._combos["precip"].currentText() == "ppt"
            assert dialog._combos["DOC"].currentText() == "doc_molL"
            assert dialog._combos["date"].currentText() == "(select…)"
            assert dialog._combos["pet"].currentText() == "(select…)"
            assert dialog._combos["q_meas"].currentText() == "(none)"
        finally:
            dialog.deleteLater()
            pump()

    def test_required_gates_ok(self, qapp: QApplication) -> None:
        """OK stays disabled while a required row is "(select…)", and
        enabling that row enables OK."""
        prefill: dict[str, str | None] = self._prefill()
        del prefill["pet"]
        dialog = self._make(qapp, prefill)
        try:
            assert not dialog._ok_button.isEnabled()
            dialog._combos["pet"].setCurrentText("evap")
            assert dialog._ok_button.isEnabled()
        finally:
            dialog.deleteLater()
            pump()

    def test_none_option(self, qapp: QApplication) -> None:
        """An optional row on "(none)" yields None in the mapping, and
        a prefilled None is applied as the row's "(none)" item."""
        dialog = self._make(qapp, self._prefill())
        try:
            dialog._combos["q_meas"].setCurrentText("(none)")
            mapping: dict[str, str | None] = dialog.mapping()
            assert mapping["q_meas"] is None
            assert mapping["DOC"] == "doc_molL"
        finally:
            dialog.deleteLater()
            pump()

        prefill: dict[str, str | None] = self._prefill()
        prefill["q_meas"] = None
        dialog = self._make(qapp, prefill)
        try:
            assert dialog._combos["q_meas"].currentText() == "(none)"
            assert dialog.mapping()["q_meas"] is None
        finally:
            dialog.deleteLater()
            pump()

    def test_mapping_roundtrip(self, qapp: QApplication) -> None:
        """A complete mapping comes back exactly as set, with every
        role key present."""
        dialog = self._make(qapp, self._prefill())
        try:
            assert dialog.mapping() == self._prefill()
        finally:
            dialog.deleteLater()
            pump()

    def test_collision_disables_ok(self, qapp: QApplication) -> None:
        """Two rows on the same column disable OK and the status line
        names the column and both roles."""
        dialog = self._make(qapp, self._prefill())
        try:
            dialog._combos["DOC"].setCurrentText("ca_molL")
            assert not dialog._ok_button.isEnabled()
            status: str = dialog._status_label.text()
            assert "ca_molL" in status
            assert "DOC" in status
            assert "Ca++" in status
        finally:
            dialog.deleteLater()
            pump()


class TestLoadData:
    """The File menu's Load data action: the file dialog, the role
    mapping dialog, the record switch, and the switches that carry
    a loaded record's column map."""

    @staticmethod
    def _days() -> pd.DatetimeIndex:
        """The twelve days every record fixture spans."""
        return pd.date_range("2020-01-01", periods=12)

    @staticmethod
    def _minimal_record(tmp_path: Path) -> Path:
        """A 12-day XLSX whose data column names match no role
        alias: only the date column auto-detects, so the dialog's
        data rows start on their sentinels and the mapping must be
        chosen by hand."""
        index: pd.DatetimeIndex = TestLoadData._days()
        frame: pd.DataFrame = pd.DataFrame(
            {
                "date": index.strftime("%Y-%m-%d"),
                "wet": np.arange(12, dtype=float),
                "degC": np.full(12, 5.0),
                "evap": np.arange(12, dtype=float) / 2.0,
            }
        )
        path: Path = tmp_path / "record.xlsx"
        frame.to_excel(path, index=False)
        return path

    @staticmethod
    def _full_record(tmp_path: Path) -> Path:
        """A 12-day XLSX carrying streamflow and the three river
        concentrations under names no role alias matches, so the
        record only loads through a complete explicit mapping."""
        index: pd.DatetimeIndex = TestLoadData._days()
        frame: pd.DataFrame = pd.DataFrame(
            {
                "date": index.strftime("%Y-%m-%d"),
                "wet": np.arange(12, dtype=float),
                "degC": np.full(12, 5.0),
                "evap": np.arange(12, dtype=float) / 2.0,
                "stream_mm": np.arange(12, dtype=float) * 3.0,
                "doc_conc": np.full(12, 1e-05),
                "ca_conc": np.full(12, 2e-03),
                "dic_conc": np.full(12, 1e-03),
            }
        )
        path: Path = tmp_path / "record.xlsx"
        frame.to_excel(path, index=False)
        return path

    @staticmethod
    def _fake_dialog(
        script: dict[str, str],
        outcome: int,
        columns_seen: list[list[str]],
        prefill_seen: list[dict[str, str | None]],
        mapping_seen: list[dict[str, str | None]],
    ) -> type[gui_data_dialog.ColumnMappingDialog]:
        """A stand-in for `ColumnMappingDialog` that records the
        columns and prefill the window handed it, applies the
        scripted combo choices, and — from `exec` — returns the
        scripted outcome while recording the mapping the window
        then reads."""

        class FakeDialog(gui_data_dialog.ColumnMappingDialog):
            def __init__(
                self,
                columns: list[str],
                network: pt.ReactionNetwork,
                prefill: dict[str, str | None],
                parent: QWidget | None = None,
            ) -> None:
                super().__init__(columns, network, prefill, parent)
                columns_seen.append(columns)
                prefill_seen.append(prefill)
                for role, value in script.items():
                    self._combos[role].setCurrentText(value)

            def exec(self) -> int:
                mapping_seen.append(self.mapping())
                return outcome

        return FakeDialog

    def test_load_data_switches_window(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A mapped XLSX loads: the dialog is prefilled by
        auto-detection, accepting it switches the window to the
        record (new inputs, path, and map), drops the old results
        and session history, and lights up the run buttons."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)

            path: Path = self._minimal_record(tmp_path)

            opens: list[tuple[str, str]] = []

            def fake_open(
                parent: object,
                title: str,
                directory: str,
                flt: str,
            ) -> tuple[str, str]:
                opens.append((title, flt))
                return (str(path), flt)

            monkeypatch.setattr(
                gui_window.QFileDialog, "getOpenFileName", fake_open
            )
            columns_seen: list[list[str]] = []
            prefill_seen: list[dict[str, str | None]] = []
            mapping_seen: list[dict[str, str | None]] = []
            monkeypatch.setattr(
                gui_window,
                "ColumnMappingDialog",
                self._fake_dialog(
                    {"precip": "wet", "temp": "degC", "pet": "evap"},
                    QDialog.DialogCode.Accepted,
                    columns_seen,
                    prefill_seen,
                    mapping_seen,
                ),
            )
            monkeypatch.setattr(gui_window, "QMessageBox", FakeMessageBox)
            FakeMessageBox.critical_calls.clear()

            win._on_load_data()
            pump()

            assert opens == [
                ("Load data", "Data files (*.csv *.xlsx)")
            ]
            assert columns_seen == [["date", "wet", "degC", "evap"]]
            assert prefill_seen == [
                {
                    "date": "date",
                    "precip": None,
                    "temp": None,
                    "pet": None,
                    "q_meas": None,
                    "DOC": None,
                    "Ca++": None,
                    "DIC": None,
                }
            ]
            expected_map: dict[str, str | None] = {
                "date": "date",
                "precip": "wet",
                "temp": "degC",
                "pet": "evap",
                "q_meas": None,
                "DOC": None,
                "Ca++": None,
                "DIC": None,
            }
            assert mapping_seen == [expected_map]
            assert win._data_path == str(path)
            assert win._column_map == expected_map
            assert win._input_data.q_meas is None
            assert win._input_data.window == (
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-12"),
            )
            assert win._history_items == []
            assert win._results is None
            assert win._figure_panel.page_count() == 0
            assert win._objective_model._table.empty
            assert win._run_action.isEnabled()
            assert win._gradients_action.isEnabled()
            assert win._line_search_action.isEnabled()
            assert win._status_label.text() == "Loaded record.xlsx."
            assert FakeMessageBox.critical_calls == []
        finally:
            close_window(win)

    def test_load_data_cancellation_keeps_session(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Declining the mapping dialog leaves the session intact:
        the inputs, the results, the path, and the map stay as
        they were."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            before: gui_data.GuiInputData = win._input_data
            before_results = win._results

            path: Path = self._minimal_record(tmp_path)

            def fake_open(
                parent: object,
                title: str,
                directory: str,
                flt: str,
            ) -> tuple[str, str]:
                return (str(path), flt)

            monkeypatch.setattr(
                gui_window.QFileDialog, "getOpenFileName", fake_open
            )
            columns_seen: list[list[str]] = []
            prefill_seen: list[dict[str, str | None]] = []
            mapping_seen: list[dict[str, str | None]] = []
            monkeypatch.setattr(
                gui_window,
                "ColumnMappingDialog",
                self._fake_dialog(
                    {},
                    QDialog.DialogCode.Rejected,
                    columns_seen,
                    prefill_seen,
                    mapping_seen,
                ),
            )
            monkeypatch.setattr(gui_window, "QMessageBox", FakeMessageBox)
            FakeMessageBox.critical_calls.clear()

            win._on_load_data()
            pump()

            assert prefill_seen[0]["date"] == "date"
            assert win._data_path is None
            assert win._column_map is None
            assert win._input_data is before
            assert win._results is before_results
            assert FakeMessageBox.critical_calls == []
        finally:
            close_window(win)

    def test_load_data_bad_file_reports_error(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A file without a date column fails before the dialog
        opens: the error is reported, the session keeps running,
        and the dialog is never constructed."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            before: gui_data.GuiInputData = win._input_data

            path: Path = tmp_path / "no_date.csv"
            pd.DataFrame(
                {
                    "wet": [1.0, 2.0, 3.0],
                    "degC": [5.0, 5.0, 5.0],
                    "evap": [0.5, 0.5, 0.5],
                }
            ).to_csv(path, index=False)

            def fake_open(
                parent: object,
                title: str,
                directory: str,
                flt: str,
            ) -> tuple[str, str]:
                return (str(path), flt)

            monkeypatch.setattr(
                gui_window.QFileDialog, "getOpenFileName", fake_open
            )
            monkeypatch.setattr(gui_window, "ColumnMappingDialog", pytest.fail)
            monkeypatch.setattr(gui_window, "QMessageBox", FakeMessageBox)
            FakeMessageBox.critical_calls.clear()

            win._on_load_data()
            pump()

            assert len(FakeMessageBox.critical_calls) == 1
            title, message = FakeMessageBox.critical_calls[0]
            assert title == "Load data"
            assert "No date column found" in message
            assert win._data_path is None
            assert win._column_map is None
            assert win._input_data is before
        finally:
            close_window(win)

    def test_load_data_corrupt_xlsx_reports_error(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A spreadsheet that cannot be parsed (a text file renamed
        ``.xlsx`` — not a valid zip) fails before the dialog opens:
        the reader converts the parser's error to a ``ValueError``,
        the slot reports it, the session keeps running, and the
        dialog is never constructed."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            before: gui_data.GuiInputData = win._input_data

            path: Path = tmp_path / "renamed_text.xlsx"
            path.write_text("this is not a zip file\n")

            def fake_open(
                parent: object,
                title: str,
                directory: str,
                flt: str,
            ) -> tuple[str, str]:
                return (str(path), flt)

            monkeypatch.setattr(
                gui_window.QFileDialog, "getOpenFileName", fake_open
            )
            monkeypatch.setattr(gui_window, "ColumnMappingDialog", pytest.fail)
            monkeypatch.setattr(gui_window, "QMessageBox", FakeMessageBox)
            FakeMessageBox.critical_calls.clear()

            win._on_load_data()
            pump()

            assert len(FakeMessageBox.critical_calls) == 1
            title, message = FakeMessageBox.critical_calls[0]
            assert title == "Load data"
            assert "Could not read the record spreadsheet" in message
            assert win._data_path is None
            assert win._column_map is None
            assert win._input_data is before
        finally:
            close_window(win)

    def test_load_data_failure_keeps_session(
        self,
        qapp: QApplication,
        input_data: gui_data.GuiInputData,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Accepting the dialog with required rows left on their
        sentinel fails in the loader: the error is reported, the
        session keeps running, and nothing on screen changed."""
        calls: list[np.ndarray] = []
        win = make_window(
            input_data,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, input_data),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            before: gui_data.GuiInputData = win._input_data
            before_results = win._results

            path: Path = tmp_path / "record.csv"
            pd.DataFrame(
                {
                    "date": TestLoadData._days()[:3].strftime("%Y-%m-%d"),
                    "foo": [1.0, 2.0, 3.0],
                }
            ).to_csv(path, index=False)

            def fake_open(
                parent: object,
                title: str,
                directory: str,
                flt: str,
            ) -> tuple[str, str]:
                return (str(path), flt)

            monkeypatch.setattr(
                gui_window.QFileDialog, "getOpenFileName", fake_open
            )
            columns_seen: list[list[str]] = []
            prefill_seen: list[dict[str, str | None]] = []
            mapping_seen: list[dict[str, str | None]] = []
            monkeypatch.setattr(
                gui_window,
                "ColumnMappingDialog",
                self._fake_dialog(
                    {},
                    QDialog.DialogCode.Accepted,
                    columns_seen,
                    prefill_seen,
                    mapping_seen,
                ),
            )
            monkeypatch.setattr(gui_window, "QMessageBox", FakeMessageBox)
            FakeMessageBox.critical_calls.clear()

            win._on_load_data()
            pump()

            assert mapping_seen[0] == {
                "date": "date",
                "precip": "(select…)",
                "temp": "(select…)",
                "pet": "(select…)",
                "q_meas": None,
                "DOC": None,
                "Ca++": None,
                "DIC": None,
            }
            assert len(FakeMessageBox.critical_calls) == 1
            title, message = FakeMessageBox.critical_calls[0]
            assert title == "Load data"
            assert (
                "The column map points the role 'precip' at the column "
                "'(select…)', which the record does not have."
            ) in message
            assert win._data_path is None
            assert win._column_map is None
            assert win._input_data is before
            assert win._results is before_results
        finally:
            close_window(win)

    def test_model_and_network_switch_reload_with_map(
        self,
        qapp: QApplication,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After a mapped load (the window's stored path and map),
        a model switch and a network switch both reload the record
        through the stored column map, keeping the model and
        network choices."""
        path: Path = self._full_record(tmp_path)
        expected_map: dict[str, str | None] = {
            "date": "date",
            "precip": "wet",
            "temp": "degC",
            "pet": "evap",
            "q_meas": "stream_mm",
            "DOC": "doc_conc",
            "Ca++": "ca_conc",
            "DIC": "dic_conc",
        }
        loaded = gui_data.load_input_data(path, column_map=expected_map)
        calls: list[np.ndarray] = []
        win = make_window(
            loaded,
            tmp_path,
            monkeypatch,
            fake_results_factory(calls, loaded),
        )
        try:
            wait_until(
                lambda: win._worker is None and win._results is not None
            )
            win._auto_run_action.setChecked(False)
            win._data_path = str(path)
            win._column_map = expected_map

            real_load = gui_data.load_input_data
            reloads: list[dict[str, object]] = []

            def spy_load(
                path: Path,
                model_key: str,
                network: pt.ReactionNetwork | None,
                column_map: dict[str, str | None] | None = None,
            ) -> gui_data.GuiInputData:
                reloads.append(
                    {
                        "path": path,
                        "model_key": model_key,
                        "network": network,
                        "column_map": column_map,
                    }
                )
                return real_load(
                    path,
                    model_key=model_key,
                    network=network,
                    column_map=column_map,
                )

            monkeypatch.setattr(gui_window, "load_input_data", spy_load)

            win._model_combo.setCurrentIndex(
                win._model_combo.findData("three_layer")
            )
            pump()

            assert len(reloads) == 1
            assert reloads[0]["path"] == path
            assert reloads[0]["model_key"] == "three_layer"
            assert reloads[0]["network"] is win._network
            assert reloads[0]["column_map"] is expected_map
            assert win._input_data.spec.key == "three_layer"
            assert win._data_path == str(path)
            assert win._column_map is expected_map

            reloads.clear()
            win._network_combo.setCurrentIndex(
                win._network_combo.findData("carbon")
            )
            pump()

            assert len(reloads) == 1
            assert reloads[0]["model_key"] == "three_layer"
            assert reloads[0]["column_map"] is expected_map
            assert reloads[0]["network"] is win._network
            carbon_fingerprint: str = gui_data.fingerprint_network(
                get_carbon_network()
            )
            assert gui_data.fingerprint_network(win._network) == (
                carbon_fingerprint
            )
            assert win._network_combo.currentData() == "carbon"
        finally:
            close_window(win)
