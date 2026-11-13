"""
The figure panel of the East River lateral-model calibration GUI.

The model's figures — the hydrology, storage, and partitioning
figures, joined by the river concentrations, river loads,
concentration vs. discharge, zone concentrations, and reaction
rates figures in full mode, the first three in hydrology-only mode,
plus a gradient job's matrix and a line search's value profile as
extra docks when those jobs have finished — each live in their own
dock window in the main window's top dock area.  The docks are
tabbed in page order, so the tab bar flips between the figures, and
a dock can be dragged out of the window into a floating window or
onto another monitor: spreading the figures across displays is the
point of the dock model.  The main window's Left / Right arrow keys
raise the previous / next dock.

Each figure is attached to a matplotlib Qt canvas.  The package's
rcParams set a publication dpi (300), which is far too large for an
interactive canvas, so the panel restates every figure at a screen
dpi (`CANVAS_DPI`) before attaching.  On a high-dpi display the
canvas rescales that dpi by the display's device pixel ratio, so
the panel restates the figure's construction-time dpi stamp to
match as well: recent versions keep the rcParams' publication dpi
in it, which would otherwise render every font three times too
large and cover the plot.  The canvas then tracks the dock's size
through matplotlib's built-in resize handling, so the visible
figure always fills its dock; because that rescaling changes how
much of the figure a point-based legend row occupies, the canvas
re-lays out the figure's below-axes legend rows on every resize,
keeping them inside the figure.
"""

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QAction, QResizeEvent
from PySide6.QtWidgets import QDockWidget, QMenu, QMainWindow

from potions.gui.plots import relayout_below_axes_legends

# The dpi the interactive canvases draw at (the package's rcParams set
# a 300 dpi for publication-quality output; restated at this dpi, a
# fifteen-inch figure is a 1500-pixel canvas instead of a 4500-pixel
# one).
CANVAS_DPI: Final[int] = 100

# The smallest height (px) a figure dock keeps: its canvas must stay
# tall enough to be readable when the user shrinks the docked figure
# area.
FIGURE_MIN_HEIGHT: Final[int] = 240

# The dock area the figure docks are built into: the window's top, so
# the docked figures sit over the table-and-parameters row exactly
# where the retired central splitter used to put the figure panel.
DOCK_AREA: Final[Qt.DockWidgetArea] = Qt.DockWidgetArea.TopDockWidgetArea


def dock_object_name(title: str, index: int) -> str:
    """
    The object name of the figure dock for the given page.

    The saved session state addresses each dock by its object name
    (Qt's `saveState` / `restoreState` layout blob records names),
    so the name must be reproducible from the page title alone: a
    rebuilt panel gives a same-titled figure the same name and the
    saved placement applies.  The slug rule matches `save_pages`'
    PNG naming — non-alphanumerics collapse to a single underscore
    and the ends are stripped — lowercased so the names read as
    identifiers.  A title that sanitizes to nothing falls back to
    the page position.

    Args:
        title: The page title.
        index: The page index, used only by the fallback name.

    Returns:
        The dock's object name.
    """
    slug: str = re.sub(r"[^0-9A-Za-z]+", "_", title).strip("_").lower()
    if not slug:
        return f"figure-dock-{index}"
    return f"figure-dock-{slug}"


# ==== Types ==== #
class FigureCanvas(FigureCanvasQTAgg):
    """
    A canvas that re-lays out its figure's below-axes legend rows
    whenever it is resized to a size it could actually be shown at.

    Matplotlib's resize handling restates the figure at the new
    canvas size, but a legend row is a fixed number of points of
    text: the figure-fraction anchor and the axes' bottom margin
    the builders measured at the construction size no longer hold
    at the new size, so the lowest row would sink past the figure's
    bottom edge.  The relayout re-measures and re-anchors the rows
    at the new size, keeping them inside the figure (and in the
    panel's fixed-size page export) at any size the dock gives the
    figure.

    The relayout runs only on a visible canvas at a shippable size.
    A tabbed dock keeps its other pages' canvases laid out at a
    degenerate tab-page size, and Qt sends them resize events at
    that size while the window is being shown; re-laying a figure
    out at 100x221 pixels asks matplotlib for a bottom margin of a
    third or more of the figure — one figure's margin passes its
    top margin outright — and running the renderer for every page
    inside the show cascade exposes a Shiboken 6.11.0 bug (its
    cached virtual-method override is stored without a reference
    count, and a canvas override detonates the freed function).
    A hidden or degenerate canvas is skipped: a raised dock gets a
    fresh resize event at its real size, which triggers the
    relayout then.
    """

    def resizeEvent(self, event: QResizeEvent) -> None:
        # The canvas class is untyped source inside the py.typed
        # matplotlib package, so both calls carry type-ignores.
        super().resizeEvent(event)  # type: ignore[no-untyped-call]
        if event.size().width() < FIGURE_MIN_HEIGHT:
            return
        if event.size().height() < FIGURE_MIN_HEIGHT:
            return
        if not self.isVisible():
            return
        # Draw once so the rows' extents are measured against a
        # canvas laid out at the new size, re-anchor them, and
        # draw the result.
        self.draw()  # type: ignore[no-untyped-call]
        relayout_below_axes_legends(self.figure)
        self.draw()  # type: ignore[no-untyped-call]


class FigureDock(QDockWidget):
    """
    One figure's dockable window.

    A titled, closable dock carrying the figure's canvas.
    `closed_from_layout` reports the user closing the dock with its
    title-bar close button — PySide6 does not expose Qt's
    `titleBarCloseButtonClicked` signal (probe fact 9), so the
    `closeEvent` override is its stand-in.  Closing the main window
    does not send close events to the docks, so the signal means
    "the user closed this dock", never "the session ended".

    Args:
        title: The page title shown in the dock's title bar.
        parent: The main window that owns the dock.
    """

    closed_from_layout = Signal()

    def __init__(self, title: str, parent: QMainWindow) -> None:
        super().__init__(title, parent)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Report the close to the panel, then perform it."""
        self.closed_from_layout.emit()
        super().closeEvent(event)


class FigurePanel:
    """
    The main window's figure area: one dock window per figure.

    The panel is not itself a widget; it manages the figure docks it
    builds into the window's top dock area and tabs them in page
    order.  The raised dock is the visible figure; the pages wrap,
    so advancing past the last dock returns to the first.  A dock
    closed with its title-bar button hides only its figure — the
    *View* menu tracks which pages remain raised.  `save_pages`
    writes every page as a PNG into a directory, as the *Save run*
    action does into the run directory's ``figures`` subfolder.

    Args:
        window: The main window the figure docks are built into.
        view_menu: The main window's *View* menu; the panel fills it
            with one checkable toggle per figure dock, rebuilt with
            every figure list.
    """

    def __init__(self, window: QMainWindow, view_menu: QMenu) -> None:
        self._window: QMainWindow = window
        self._view_menu: QMenu = view_menu
        # The page-order lists stay parallel: docks[i] carries
        # canvases[i], is titled titles[i], and is toggled by
        # view_actions[i].
        self._docks: list[FigureDock] = []
        self._canvases: list[FigureCanvasQTAgg] = []
        self._titles: list[str] = []
        self._view_actions: list[QAction] = []
        # The raised page, or -1 while no figures are loaded.
        self._current_index: int = -1
        # Bumped with every rebuild.  A rebuild schedules its page
        # re-assertion (see `_reassert_page`) tagged with this
        # generation, and a scheduled call drops itself when a newer
        # rebuild has since replaced the docks.
        self._rebuild_generation: int = 0
    # =================== #

    # ==== Public ==== #
    def set_figures(self, figures: Sequence[tuple[str, Figure]]) -> None:
        """
        Replace the figure docks with docks for the given figures.

        Every figure is restated at `CANVAS_DPI` — its dpi and the
        construction-time stamp the high-dpi rescale bases itself
        on — attached to a fresh canvas, and wrapped in a titled
        dock.  The previously raised page is kept when its title
        survives the rebuild, at that title's new position, so a
        rebuild (e.g. after a plot-selection change or the line
        search adding its profile page) does not move the user to a
        different figure; a title that is gone falls back to the
        previous index clamped into the new range.  Qt settles a
        fresh tabify chain on a later event-loop pass and raises the
        first dock there, so the chosen page is re-asserted once
        those queued layout events have run (see `_reassert_page`).

        The old docks are removed *before* the new ones are built:
        Qt's dock state addresses docks by object name, so a
        same-titled figure must have its old dock's name free to
        take over the old dock's placement.  The *View* menu's
        toggles are detached before the docks they mirror die, then
        rebuilt: each new dock gets a checked, checkable action that
        shows and hides the dock, and whose checkbox the dock's own
        title-bar close unchecks.

        Args:
            figures: The (page title, figure) pairs, in page order.
        """
        self._rebuild_generation += 1
        previous_index: int = self._current_index
        previous_title: str | None = (
            self._titles[previous_index]
            if 0 <= previous_index < len(self._titles)
            else None
        )
        # Detach the old toggles before the docks: removing them
        # first keeps a `toggled` signal from touching a dying dock
        # during the teardown below.
        for stale_action in self._view_actions:
            self._view_menu.removeAction(stale_action)
            stale_action.deleteLater()
        self._view_actions.clear()
        self._remove_docks()

        for index, (title, fig) in enumerate(figures):
            fig.set_dpi(CANVAS_DPI)
            # `set_dpi` leaves the construction-time `_original_dpi`
            # stamp in place, and recent matplotlibs keep the
            # rcParams' publication dpi in that stamp; the canvas'
            # high-dpi hook rescales the figure by the display ratio
            # times the stamp, so it is restated to the canvas dpi
            # here as well (the module docstring explains the effect
            # of a stale stamp).  The stamp is private, hence the
            # ignore.
            fig._original_dpi = CANVAS_DPI  # type: ignore[attr-defined]
            # The canvas is untyped source inside the py.typed
            # matplotlib package, so the constructor call carries a
            # type-ignore.
            canvas: FigureCanvas = FigureCanvas(
                fig
            )  # type: ignore[no-untyped-call]
            dock: FigureDock = FigureDock(title, self._window)
            dock.setObjectName(dock_object_name(title, index))
            dock.setMinimumHeight(FIGURE_MIN_HEIGHT)
            dock.setWidget(canvas)
            dock.visibilityChanged.connect(self._visibility_handler(index))
            self._window.addDockWidget(DOCK_AREA, dock)
            action: QAction = QAction(title, self._window)
            action.setCheckable(True)
            # Checked *before* the connections: `setChecked` itself
            # emits `toggled`, and the pair must not run while it is
            # still half-wired.
            action.setChecked(True)
            action.toggled.connect(self._toggle_handler(dock))
            dock.closed_from_layout.connect(self._close_handler(action))
            self._view_menu.addAction(action)
            self._view_actions.append(action)
            self._docks.append(dock)
            self._canvases.append(canvas)
            self._titles.append(title)

        # Tabify in page order.  `tabifyDockWidget(first, second)`
        # raises the second dock, so the chain alone would leave the
        # last figure raised; the explicit activation below picks
        # the page to show.  A lone dock gets no tab bar — fine.
        for tabbed in range(1, len(self._docks)):
            self._window.tabifyDockWidget(
                self._docks[tabbed - 1], self._docks[tabbed]
            )

        if self._docks:
            target: int = min(previous_index, len(self._docks) - 1)
            if target < 0:
                target = 0
            if previous_title is not None and previous_title in self._titles:
                target = self._titles.index(previous_title)
            self._docks[target].raise_()
            self._current_index = target
            generation: int = self._rebuild_generation
            QTimer.singleShot(
                0, lambda: self._reassert_page(target, generation)
            )

    def docks(self) -> list[FigureDock]:
        """
        The figure docks, in page order.

        The *View* menu's per-figure toggles and the saved dock
        layout build on this list; the copy keeps the panel's own
        list private.

        Returns:
            A list of the panel's current figure docks.
        """
        return list(self._docks)

    def save_pages(self, directory: Path) -> list[Path]:
        """
        Save every page's figure as a PNG into the given directory.

        Each figure is saved as plotted (the mode's own figures,
        and the gradient and line search pages when those jobs have
        finished), at `CANVAS_DPI`, named after the page's title:
        anything outside ``0-9A-Za-z`` collapses to a single
        underscore and the leading and trailing underscores are
        stripped (a title that sanitizes to nothing falls back to
        the page's position).

        Args:
            directory: The directory to write the PNGs into (the
                run directory's ``figures`` subfolder); created
                when missing.

        Returns:
            The paths of the written files, in page order.
        """
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, canvas in enumerate(self._canvases):
            name: str = re.sub(r"[^0-9A-Za-z]+", "_", self._titles[index])
            name = name.strip("_")
            if not name:
                name = f"figure_{index + 1}"
            path: Path = directory / f"{name}.png"
            canvas.figure.savefig(path, dpi=CANVAS_DPI)
            paths.append(path)
        return paths

    def show_page(self, index: int) -> None:
        """
        Show the page at the given index.

        Args:
            index: The page index, clamped into the valid range.
        """
        if not self._docks:
            return
        target: int = min(max(index, 0), len(self._docks) - 1)
        self._docks[target].raise_()
        self._current_index = target

    def next_page(self) -> None:
        """Show the next page (wrapping to the first)."""
        if len(self._docks) > 1:
            self.show_page((self._current_index + 1) % len(self._docks))

    def previous_page(self) -> None:
        """Show the previous page (wrapping to the last)."""
        if len(self._docks) > 1:
            self.show_page((self._current_index - 1) % len(self._docks))

    def page_count(self) -> int:
        """
        The number of figure docks in the panel.

        Returns:
            The number of loaded figures.
        """
        return len(self._docks)

    def current_page(self) -> int:
        """
        The index of the raised page.

        Returns:
            The page index, or -1 when no pages are loaded.
        """
        return self._current_index
    # =================== #

    # ==== Internal ==== #
    def _remove_docks(self) -> None:
        """Take every figure dock out of the window and delete it."""
        for dock in self._docks:
            self._window.removeDockWidget(dock)
            # Deleting the dock deletes the canvas it carries, so
            # the canvases need no removal of their own.
            dock.deleteLater()
        self._docks.clear()
        self._canvases.clear()
        self._titles.clear()
        self._current_index = -1

    def _reassert_page(self, page: int, generation: int) -> None:
        """
        Re-raise the page a rebuild chose, unless a newer rebuild replaced it.

        `set_figures` raises the chosen page synchronously, but Qt
        finalizes the fresh tabify chain on a later event-loop pass
        and raises the first dock there; that dock's visibility
        handler then records page 0 as the current page, discarding
        the choice.  Scheduling this call with a zero-delay timer
        makes it run after those queued layout events and restore
        the chosen page.  The generation check drops the call when
        `set_figures` has rebuilt the docks again in the meantime,
        so a stale re-assertion can never raise a dead dock or
        override a newer rebuild's page.

        Args:
            page: The page index the rebuild chose to show.
            generation: The rebuild generation the call was scheduled for.
        """
        if generation != self._rebuild_generation:
            return
        self.show_page(page)

    def _visibility_handler(self, page: int) -> Callable[[bool], None]:
        """
        Build the `visibilityChanged` slot for one figure dock.

        A dock becoming visible is the tab bar's (or a
        `show_page` call's) way of announcing the new current page;
        becoming hidden is the flip side of another dock's showing
        and is ignored.  The factory binds the page index the dock
        was built for.

        Args:
            page: The dock's page index.

        Returns:
            The slot to connect to the dock's `visibilityChanged`.
        """
        def on_visibility(visible: bool) -> None:
            if visible:
                self._current_index = page

        return on_visibility

    def _toggle_handler(self, dock: FigureDock) -> Callable[[bool], None]:
        """
        Build the *View* menu toggle slot for one figure dock.

        The menu checkbox mirrors the dock's open state: checking
        the action shows the dock (its tab returns to the strip's
        tab bar), unchecking it hides it.  The dock's own title-bar
        close already hides the dock, and the follow-up
        `toggled(False)` call on the hidden dock is a no-op.

        Args:
            dock: The dock the menu action toggles.

        Returns:
            The slot to connect to the action's `toggled`.
        """
        def on_toggled(visible: bool) -> None:
            dock.setVisible(visible)

        return on_toggled

    def _close_handler(self, action: QAction) -> Callable[[], None]:
        """
        Build the `closed_from_layout` slot for one *View* toggle.

        The dock's title-bar close button (through
        `FigureDock.closeEvent`, the stand-in for Qt's unexposed
        close-button signal) unchecks the paired menu action, so the
        menu keeps reporting which figures are open even when the
        user closes docks with their own button.

        Args:
            action: The menu action paired with the closing dock.

        Returns:
            The slot to connect to that dock's `closed_from_layout`.
        """
        def on_closed() -> None:
            action.setChecked(False)

        return on_closed
    # =================== #
# =============== #
