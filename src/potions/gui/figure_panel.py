"""
The figure panel of the East River lateral-model calibration GUI.

Holds the model's three figures (streamflow and river concentrations,
zone concentrations and storages, and mineral reaction rates) in a
single panel: only one figure is visible at a time, and the
previous / next buttons of the panel's header — or the main window's
Left / Right arrow keys, which are wired to the same handlers — flip
between the pages.

Each figure is attached to a matplotlib Qt canvas.  The package's
rcParams set a publication dpi (300), which is far too large for an
interactive canvas, so the panel restates every figure at a screen
dpi (`CANVAS_DPI`) before attaching.  The canvas then tracks the
panel size through matplotlib's built-in resize handling, so the
visible figure always fills the panel.
"""

from collections.abc import Sequence
from typing import Final

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# The dpi the interactive canvases draw at (the package's rcParams set
# a 300 dpi for publication-quality output; restated at this dpi, a
# fifteen-inch figure is a 1500-pixel canvas instead of a 4500-pixel
# one).
CANVAS_DPI: Final[int] = 100


# ==== Types ==== #
class FigurePanel(QWidget):
    """
    The single figure panel of the main window.

    A stacked widget of matplotlib canvases with a header carrying
    the current page's title, a ``n / N`` counter, and previous /
    next buttons.  The pages wrap around, so the last page's next
    button returns to the first.

    Args:
        parent: The parent widget; defaults to None.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._titles: list[str] = []
        self._canvases: list[FigureCanvasQTAgg] = []

        self._title_label: QLabel = QLabel("", self)
        self._prev_button: QPushButton = QPushButton("◀", self)
        self._page_label: QLabel = QLabel("", self)
        self._next_button: QPushButton = QPushButton("▶", self)
        self._prev_button.setFixedSize(32, 24)
        self._next_button.setFixedSize(32, 24)
        self._prev_button.setToolTip("Previous figure (Left arrow)")
        self._next_button.setToolTip("Next figure (Right arrow)")
        self._prev_button.clicked.connect(self.previous_page)
        self._next_button.clicked.connect(self.next_page)

        self._stack: QStackedWidget = QStackedWidget(self)
        self._stack.currentChanged.connect(self._refresh_header)

        header: QHBoxLayout = QHBoxLayout()
        header.addWidget(self._title_label, 1)
        header.addWidget(self._prev_button)
        header.addWidget(self._page_label)
        header.addWidget(self._next_button)

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self._stack, 1)

        self._refresh_header()
    # =================== #

    # ==== Public ==== #
    def set_figures(self, figures: Sequence[tuple[str, Figure]]) -> None:
        """
        Replace the panel's pages with the given titled figures.

        Every figure is restated at `CANVAS_DPI` and attached to a
        fresh canvas.  The previously visible page is kept when it
        still exists, so a rebuild (e.g. after a plot-selection
        change) does not move the user to a different figure.

        Args:
            figures: The (page title, figure) pairs, in page order.
        """
        previous: int = self._stack.currentIndex()
        for old_canvas in self._canvases:
            self._stack.removeWidget(old_canvas)
            old_canvas.deleteLater()
        self._canvases.clear()
        self._titles = [title for title, _ in figures]

        for title, fig in figures:
            fig.set_dpi(CANVAS_DPI)
            # matplotlib's Qt canvas is untyped source inside the
            # py.typed matplotlib package, so the constructor call
            # carries the only type-ignore in this module.
            canvas: FigureCanvasQTAgg = FigureCanvasQTAgg(
                fig
            )  # type: ignore[no-untyped-call]
            self._stack.addWidget(canvas)
            self._canvases.append(canvas)

        if self._canvases:
            target: int = previous if previous >= 0 else 0
            self._stack.setCurrentIndex(min(target, len(self._canvases) - 1))
            self._fit_current()
        # The currentChanged signal fired during the addWidget loop,
        # before the last canvases were recorded, and the final
        # setCurrentIndex is often a no-op: refresh the header
        # explicitly so it always matches the new page list.
        self._refresh_header()

    def show_page(self, index: int) -> None:
        """
        Show the page at the given index.

        Args:
            index: The page index, clamped into the valid range.
        """
        if not self._canvases:
            return
        self._stack.setCurrentIndex(min(max(index, 0), len(self._canvases) - 1))

    def next_page(self) -> None:
        """Show the next page (wrapping to the first)."""
        if len(self._canvases) > 1:
            self._stack.setCurrentIndex(
                (self._stack.currentIndex() + 1) % len(self._canvases)
            )

    def previous_page(self) -> None:
        """Show the previous page (wrapping to the last)."""
        if len(self._canvases) > 1:
            self._stack.setCurrentIndex(
                (self._stack.currentIndex() - 1) % len(self._canvases)
            )

    def page_count(self) -> int:
        """
        The number of pages in the panel.

        Returns:
            The number of loaded figures.
        """
        return len(self._canvases)

    def current_page(self) -> int:
        """
        The index of the visible page.

        Returns:
            The page index, or -1 when no pages are loaded.
        """
        return self._stack.currentIndex()
    # =================== #

    # ==== Internal ==== #
    def _refresh_header(self) -> None:
        """Update the header's title, page counter, and button states."""
        num_pages: int = len(self._canvases)
        index: int = self._stack.currentIndex()
        if num_pages == 0 or index < 0:
            self._title_label.setText("")
            self._page_label.setText("")
        else:
            self._title_label.setText(self._titles[index])
            self._page_label.setText(f"{index + 1} / {num_pages}")
        self._prev_button.setEnabled(num_pages > 1)
        self._next_button.setEnabled(num_pages > 1)

    def _fit_current(self) -> None:
        """
        Fit the visible canvas to the panel.

        Resizing the canvas triggers matplotlib's own resize handling
        (which restates the figure at the canvas size and schedules a
        draw), so the figure fills the panel without a custom render
        pass.  A zero-size panel (the layout has not run yet) is
        skipped; the canvas fits itself through the resize events it
        receives once the window is shown.
        """
        if not self._canvases:
            return
        width: int = self._stack.width()
        height: int = self._stack.height()
        if width <= 0 or height <= 0:
            return
        index: int = self._stack.currentIndex()
        if index < 0:
            return
        canvas: FigureCanvasQTAgg = self._canvases[index]
        canvas.resize(width, height)
    # =================== #
# =============== #
