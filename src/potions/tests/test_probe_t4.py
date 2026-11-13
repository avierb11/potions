"""Temporary probe for Task 4 dock-state semantics (deleted after use)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTabBar

import potions as pt
from ..gui import data as gui_data
from ..gui import window as gui_window
from .test_gui import input_data  # noqa: F401  (fixture re-export)
from .test_gui_qt import (  # noqa: F401  (fixture re-export + helpers)
    _close_window,
    _fake_results_factory,
    _make_window,
    _pump,
    _wait_until,
    qapp,
)


def _heights(win: gui_window.MainWindow) -> tuple[float, float]:
    panel: gui_window.FigurePanel = win._figure_panel
    dock = panel.docks()[panel.current_page()]
    return float(dock.height()), float(win.centralWidget().height())


def _tab_info(win: gui_window.MainWindow) -> tuple[int, list[str]]:
    bars = [bar for bar in win.findChildren(QTabBar) if bar.isVisible()]
    if len(bars) != 1:
        return (len(bars), [])
    bar = bars[0]
    return bar.currentIndex(), [
        bar.tabText(i) for i in range(bar.count())
    ]


def _bar_report(win: gui_window.MainWindow) -> None:
    for bar in win.findChildren(QTabBar):
        print(
            "BAR",
            "visible:", bar.isVisible(),
            "count:", bar.count(),
            "geom:", bar.geometry(),
            "parent:", type(bar.parent()).__name__,
            "texts:", [bar.tabText(i) for i in range(bar.count())],
        )


def test_probe_save_restore(
    qapp: QApplication,
    input_data: gui_data.GuiInputData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure saveState/restoreState behavior on the real window."""
    calls: list[np.ndarray] = []
    real_save = gui_window.save_model_run
    saved_dirs: list[Path] = []

    def fake_save(
        model: pt.Model,
        hydro: pt.HydroModelResults,
        rt: pt.RtModelResults,
        results_dir: Path | None = None,
    ) -> Path:
        run_dir = real_save(model, hydro, rt, results_dir)
        saved_dirs.append(run_dir)
        return run_dir

    monkeypatch.setattr(gui_window, "save_model_run", fake_save)
    win1 = _make_window(
        input_data,
        tmp_path,
        monkeypatch,
        _fake_results_factory(calls, input_data),
    )
    _wait_until(lambda: win1._worker is None and win1._results is not None)
    win1.show()
    _pump()
    win1.setGeometry(0, 0, 1100, 800)
    _pump()
    panel1 = win1._figure_panel
    print("PROBE natural:", _heights(win1), "minHint:", win1.centralWidget().minimumSizeHint().height(), "dockmin:", panel1.docks()[0].minimumHeight())
    win1.resizeDocks(
        [panel1.docks()[0]], [200], Qt.Orientation.Vertical
    )
    _pump()
    h1, c1 = _heights(win1)
    print("PROBE request200:", (h1, c1), "ratio:", h1 / (h1 + c1))
    print("PROBE all dock heights:", [d.height() for d in panel1.docks()])
    blob_a = bytes(win1.saveState())
    blob_b = bytes(win1.saveState())
    print("PROBE deterministic:", blob_a == blob_b, "len:", len(blob_a))
    print("PROBE tab info:", _tab_info(win1))
    win1._on_save_run()
    print("PROBE saved dir:", saved_dirs, "status:", win1._status_label.text())
    _close_window(win1)

    # Restore path A: window auto-restores the saved run; then
    # restoreState(blob) while shown, then resize.
    win2 = gui_window.MainWindow(input_data)
    _pump()
    panel2 = win2._figure_panel
    print("PROBE win2 docks before show:", len(panel2.docks()))
    ok2 = win2.restoreState(blob_a)
    _pump()
    win2.show()
    _pump()
    win2.resize(1100, 800)
    _pump()
    h2, c2 = _heights(win2)
    print("PROBE win2 restored(shown-first):", ok2, (h2, c2), "ratio:", h2 / (h2 + c2))
    _bar_report(win2)
    _pump()
    _bar_report(win2)
    print("PROBE win2 areas:", [
        (win2.dockWidgetArea(d), d.isFloating(), d.isVisible())
        for d in panel2.docks()
    ])
    print("PROBE win2 tabs:", _tab_info(win2))
    _close_window(win2)

    # Restore path C (remove-first): remove all docks from the layout,
    # then restoreState, then show.
    win5 = gui_window.MainWindow(input_data)
    _pump()
    panel5 = win5._figure_panel
    for d in panel5.docks():
        win5.removeDockWidget(d)
    _pump()
    ok5 = win5.restoreState(blob_a)
    _pump()
    win5.show()
    _pump()
    win5.setGeometry(0, 0, 1100, 800)
    _pump()
    h5, c5 = _heights(win5)
    print("PROBE win5 remove-first:", ok5, (h5, c5), "ratio:", h5 / (h5 + c5))
    _bar_report(win5)
    _pump()
    _bar_report(win5)
    print("PROBE win5 areas:", [
        (win5.dockWidgetArea(d), d.isFloating(), d.isVisible())
        for d in panel5.docks()
    ])
    print("PROBE win5 tabs:", _tab_info(win5))
    print("PROBE win5 dock heights:", [d.height() for d in panel5.docks()])
    _close_window(win5)

    # resizeDocks(380) while hidden, then show: the default-layout path.
    win4 = _make_window(
        input_data,
        tmp_path,
        monkeypatch,
        _fake_results_factory(calls, input_data),
    )
    _wait_until(
        lambda: win4._worker is None
        and win4._results is not None
        and win4._figure_panel.docks() != []
    )
    win4.resizeDocks(
        [win4._figure_panel.docks()[0]], [380], Qt.Orientation.Vertical
    )
    win4.show()
    _pump()
    win4.setGeometry(0, 0, 1100, 800)
    _pump()
    h4, c4 = _heights(win4)
    print("PROBE win4 hidden-resize380:", (h4, c4), "ratio:", h4 / (h4 + c4))
    _close_window(win4)
