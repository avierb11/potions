"""
The parameter panel of the East River calibration GUI.

Holds the selected model's adjustable parameters in the panel below
the figure: the hydrology parameters of every zone (blue-themed), the
single catchment-proportion parameter (when the model has one), the
reactive-transport parameters of every zone (red-themed, with a header
per mineral in the reactive zones), and the calibrated initial state.
The zones, their labels and colors, and their chemistry all come from
the selected `potions.gui.model_specs.ModelSpec`, so switching the
model rebuilds the panel in place (see `ParamPanel.reload`).

Every parameter whose bounds are distinct is a spin-box cell whose
range can be re-edited by double-clicking the cell; the value's
displayed precision follows the current range (see `bound_decimals`).
A parameter whose bounds are identical is pinned and shown as a
read-only cell; double-clicking the cell opens a dialog that gives
it a range, converting it into a free cell.

In a themed group the cells form a compact grid: each cell shows
the parameter's symbol to the left of its value (or pinned value),
and is drawn with a thin border and an alternating row shading, so
the grid reads as a table of cells.  Every group's title is
left-aligned, so the titles line up at the panel's left edge.

The "Plots" group carries the controls that select what the figures
draw: the river-concentration figure's streamflow toggle and river
species (shared with the zone-concentration figure), the zone
checkboxes (shared with the storage figure), the reaction-rate
figure's single mineral, and the two date pickers that set the
plotting window of every figure.  It emits `selection_changed`
whenever one of the controls changes.

The final group, "History", lists the previous runs in a dropdown
(the saved run directories and the session's in-memory runs, newest
first); the *Overlay previous results* checkbox asks the window to
draw the selected run's results faintly on the current figures, and
the *Use these parameters* button asks it to restore that run's
parameters to the panel without a re-run.
"""

from typing import Final

from PySide6.QtCore import QDate, QEvent, QObject, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import potions as pt

from potions.gui.common import (
    HYDRO_THEME,
    MINERAL_PARAMS,
    RT_DIM_PARAMS,
    RT_THEME,
    Theme,
    bound_decimals,
    param_label,
    param_symbol,
)
from potions.gui.data import (
    DEFAULT_WINDOW_END,
    DEFAULT_WINDOW_START,
    GuiInputData,
    PlotSelection,
)

# ==== Constants ==== #
# The arrows of a group's collapse button, expanded or not.
ARROW_EXPANDED: Final[str] = "▾"
ARROW_COLLAPSED: Final[str] = "▸"

# The color chip left of a zone subheader, tinted with the zone's
# color from the figure palette.
ZONE_CHIP: Final[str] = "■"

# The range of the bound-edit dialog's spin boxes: wide enough for
# any of the model's bounds (a few thousands at most), with the
# decimals of the saved parameters.csv files.
BOUND_RANGE_MIN: Final[float] = -1e6
BOUND_RANGE_MAX: Final[float] = 1e6
BOUND_DECIMALS: Final[int] = 6

# The number of columns in the parameter groups' cell grids: each
# parameter cell takes one column (a row holds five, the most any
# zone carries), and a full-width header (a zone or mineral name)
# spans all of them.
GRID_COLUMNS: Final[int] = 5

# The decoration of a themed group's cell grid, keyed by the
# theme's name: (the band of the even cell rows, the band of the
# odd cell rows, the color of the 1px border that draws the
# grid).  The two bands are close light shades — one near white,
# one a slightly darker variant of the theme's light background —
# tinted to match the group (blue-gray for the hydro groups, warm
# for the reactive-transport groups).  A new `Theme` must be
# registered here.
GRID_STYLES: Final[dict[str, tuple[str, str, str]]] = {
    HYDRO_THEME.name: ("#F7FAFC", "#E2EBF5", "#C6D5E4"),
    RT_THEME.name: ("#FEF9F8", "#F6E1DD", "#E2C6C1"),
}


# =================== #


# ==== Widgets ==== #
class FreeCell(QWidget):
    """
    A single free-parameter cell: the parameter's symbol to the
    left of a spin box holding its value.

    The spin box is frameless: the group that hosts the cell draws
    the grid (a thin border and an alternating row shading, see
    `ThemedGroup`), which is the cell's only frame.  The spin box
    is clamped to the cell's (possibly user-edited) range, and its
    decimal precision and step width follow the range
    (`bound_decimals`).  The symbol carries the parameter's full
    label as its tooltip, so the cell shows only the symbol while
    the full name is available on hover.  Double-clicking anywhere
    on the cell emits `edit_requested`, which the panel answers by
    opening the dialog that edits both ends of the range.

    Signals:
        value_changed: Emitted with the parameter key whenever the
            value changes.
        edit_requested: Emitted with the parameter key when the
            cell is double-clicked.

    Args:
        key: The parameter's bounds key.
        symbol: The parameter's short display symbol.
        label: The parameter's full label, shown as the cell's
            tooltip.
        bounds: The parameter's current (min, max) range.
        value: The parameter's current value, clamped into the range.
        theme: The theme of the group the cell belongs to, or None
            for an unthemed group; the symbol is tinted with it.
        parent: The parent widget; defaults to None.
    """

    value_changed = Signal(str)
    edit_requested = Signal(str)

    def __init__(
        self,
        key: str,
        symbol: str,
        label: str,
        bounds: tuple[float, float],
        value: float,
        theme: Theme | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._key: str = key
        self._bounds: tuple[float, float] = bounds
        self._value: float = value

        # The symbol to the left of the value, the value right-
        # aligned to the cell's right edge; the stretch between
        # them keeps the two pinned to opposite ends.
        layout: QHBoxLayout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._symbol: QLabel = QLabel(symbol)
        self._symbol.setToolTip(label)
        self._symbol.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        if theme is not None:
            self._symbol.setStyleSheet(f"color: {theme.accent};")
        layout.addWidget(self._symbol)
        layout.addStretch(1)

        # Frameless: the cell's border, drawn by the group, is the
        # cell's only frame, so the grid lines stay continuous.
        self._spin: QDoubleSpinBox = QDoubleSpinBox()
        self._spin.setToolTip(label)
        self._spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._spin.setStyleSheet(
            "border: none; background: transparent;"
            "QLineEdit { background: transparent; }"
        )
        self._spin.valueChanged.connect(self._on_value_edited)
        layout.addWidget(self._spin)

        self._line_edit: QLineEdit | None = self._spin.lineEdit()
        if self._line_edit is not None:
            # A double-click on the line edit never reaches the
            # cell's double-click handler, so the line edit is
            # watched for one (see `eventFilter`).
            self._line_edit.installEventFilter(self)
        # The symbol label swallows double-clicks the same way
        # (its selectable text eats the click), so it is watched
        # too (see `eventFilter`).
        self._symbol.installEventFilter(self)

        self.set_bounds(*bounds)

    def value(self) -> float:
        """The parameter's current value, in the current range."""
        return self._value

    def set_value(self, value: float) -> None:
        """
        Clamp `value` into the current range and move the spin box.

        The spin box's signals are blocked while the value moves, so
        this does not re-emit `value_changed`; the caller re-emits
        when the change came from outside the cell (an edited
        range, a restored state).

        Args:
            value: The value to set, clamped into the range.
        """
        lo, hi = self._bounds
        self._value = min(max(value, lo), hi)
        self._spin.blockSignals(True)
        self._spin.setValue(self._value)
        self._spin.blockSignals(False)

    def set_bounds(self, lo: float, hi: float) -> None:
        """
        Replace the range, keeping the current (clamped) value.

        The spin box's range, decimal precision, and step width all
        follow the new bounds.  Like `set_value`, the spin box's
        signals are blocked, so the caller re-emits the resulting
        changes.

        Args:
            lo: The new lower bound.
            hi: The new upper bound.
        """
        self._bounds = (lo, hi)
        decimals: int = bound_decimals(lo, hi)
        self._spin.blockSignals(True)
        self._spin.setRange(lo, hi)
        self._spin.setDecimals(decimals)
        self._spin.setSingleStep(10.0 ** (-decimals))
        self._value = min(max(self._value, lo), hi)
        self._spin.setValue(self._value)
        self._spin.blockSignals(False)

    def _on_value_edited(self, value: float) -> None:
        """Record a user edit of the spin box and emit `value_changed`."""
        self._value = value
        self.value_changed.emit(self._key)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """
        Open the range dialog when the spin box's line edit or the
        symbol label is double-clicked (a click in either never
        reaches the cell's double-click handler).

        Args:
            watched: The object the event was delivered to.
            event: The event itself.

        Returns:
            Always False: the event keeps propagating.
        """
        if (
            watched is self._line_edit
            or watched is self._symbol
        ) and event.type() == QEvent.Type.MouseButtonDblClick:
            self.edit_requested.emit(self._key)
        return False

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """
        Open the range dialog when any part of the cell is
        double-clicked, then forward the event.
        """
        self.edit_requested.emit(self._key)
        super().mouseDoubleClickEvent(event)


def compact_value(value: float) -> str:
    """
    A compact rendering of a pinned value: four decimals (the
    saved ``parameters.csv`` precision) with the trailing zeros
    stripped, so that a pinned value reads as "0", "1", or "5000"
    rather than "0.0000", "1.0000", or "5000.0000".

    Args:
        value: The pinned value.

    Returns:
        The compact text to show for the value.
    """
    return f"{value:.4f}".rstrip("0").rstrip(".")


class PinnedCell(QWidget):
    """
    A read-only cell for a pinned parameter, whose bounds are
    identical so there is nothing to adjust.

    The value is shown in a compact, disabled (grayed) label to
    the right of the symbol, rather than in a spin box: a
    disabled spin box's frame and buttons take up most of a narrow
    grid cell, so the fixed value is plain text (see
    `compact_value`).  The symbol carries the parameter's full
    label as its tooltip.  Double-clicking the cell emits
    `unlock_requested`, which the panel answers by opening the
    range dialog and converting the cell into a free cell.

    Like a free cell, the group that hosts it draws the grid.

    Args:
        key: The parameter's bounds key.
        symbol: The parameter's short display symbol.
        label: The parameter's full label, shown as the cell's
            tooltip.
        value: The pinned value, shown in gray.
        parent: The parent widget; defaults to None.
    """

    unlock_requested = Signal(str)

    def __init__(
        self,
        key: str,
        symbol: str,
        label: str,
        value: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._key: str = key

        # The symbol to the left of the value, the value right-
        # aligned to the cell's right edge; the stretch between
        # them keeps the two pinned to opposite ends.
        layout: QHBoxLayout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self._symbol: QLabel = QLabel(symbol)
        self._symbol.setToolTip(label)
        self._symbol.setStyleSheet("color: gray;")
        layout.addWidget(self._symbol)
        layout.addStretch(1)

        # A disabled label, not a spin box (see the class
        # docstring).  It is disabled so that a double-click on
        # the value is forwarded to this cell, as Qt does for
        # disabled children.  The label takes the cell's remaining
        # width, and its right alignment keeps the value at the
        # cell's right edge, as in a free cell.
        self._value: QLabel = QLabel(compact_value(value))
        self._value.setStyleSheet("color: gray;")
        self._value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._value.setEnabled(False)
        layout.addWidget(self._value)

        self.setToolTip(
            "Pinned to a single value. Double-click to give this "
            "parameter a range."
        )

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """
        Emit `unlock_requested` for this cell, then forward the
        event (a double-click on the disabled value label is
        forwarded here by Qt).
        """
        self.unlock_requested.emit(self._key)
        super().mouseDoubleClickEvent(event)



class BoldHeader(QLabel):
    """
    A bold subheader inside a group: the name of a mineral or of a
    plot-selection section.

    Args:
        text: The header's text, rendered bold.
        top_margin: The top margin (pixels) above the header.
        parent: The parent widget; defaults to None.
    """

    def __init__(
        self,
        text: str,
        top_margin: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setText(f"<b>{text}</b>")
        self.setContentsMargins(4, top_margin, 4, 0)


def labeled_date_edit(
    label: str,
    value: str,
    minimum: str,
    maximum: str,
) -> tuple[QWidget, QDateEdit]:
    """
    A row widget of a label and a date picker showing `value`.

    The *Plots* group's plotting-period pickers are built this way,
    so a picker's label and date edit share one cell of the group's
    two-column grid instead of taking a cell each.  The picker spans
    the loaded record's window (from `minimum` to `maximum`), so a
    date outside the record cannot be selected.

    Args:
        label: The label text, left of the date edit.
        value: The initial date, as an ISO date.
        minimum: The earliest selectable date, as an ISO date.
        maximum: The latest selectable date, as an ISO date.

    Returns:
        The row widget and the date edit it holds.
    """
    row: QWidget = QWidget()
    layout: QHBoxLayout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(QLabel(label))
    edit: QDateEdit = QDateEdit()
    edit.setCalendarPopup(True)
    edit.setDisplayFormat("yyyy-MM-dd")
    edit.setDateRange(
        QDate.fromString(minimum, Qt.DateFormat.ISODate),
        QDate.fromString(maximum, Qt.DateFormat.ISODate),
    )
    edit.setDate(QDate.fromString(value, Qt.DateFormat.ISODate))
    layout.addWidget(edit)
    return row, edit


class ZoneHeader(QLabel):
    """
    The bold subheader of a zone's parameters: the zone's color chip
    (the zone's color from the model spec) followed by its display
    name.

    Args:
        zone: The zone's name; kept for the group's bookkeeping.
        label: The zone's display name.
        color: The zone's color from the model spec's palette.
        parent: The parent widget; defaults to None.
    """

    def __init__(
        self,
        zone: str,
        label: str,
        color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._zone: str = zone
        self.setText(
            f'<span style="color: {color};">{ZONE_CHIP}</span> '
            f"<b>{label}</b>"
        )
        self.setContentsMargins(4, 4, 0, 0)


class BoundsDialog(QDialog):
    """
    The range-edit dialog: two spin boxes prefilled with the
    current lower and upper bounds, an optional hint line, a "make
    static" checkbox, and OK / Cancel buttons.

    With the checkbox unchecked, OK stays disabled until the two
    bounds form a strict range (lower < upper); the check runs on
    every change of either spin box.  Checking the box disables
    both spin boxes and sets them to the parameter's current value
    (the backed-up values are restored when the box is unchecked
    again), so the accepted result is a degenerate range that pins
    the parameter at that value.

    Args:
        title: The dialog's window title.
        lo: The current lower bound, prefilled in the lower spin box.
        hi: The current upper bound, prefilled in the upper spin box.
        hint: An optional gray hint line shown above the spin boxes.
        value: The parameter's current value; the static option
            pins the parameter at this value.
        parent: The parent widget; defaults to None.
    """

    def __init__(
        self,
        title: str,
        lo: float,
        hi: float,
        hint: str | None,
        value: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(340)
        self._value: float = value
        self._backed_up: tuple[float, float] = (lo, hi)

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        if hint is not None:
            hint_label: QLabel = QLabel(hint)
            hint_label.setWordWrap(True)
            hint_label.setStyleSheet("color: gray;")
            layout.addWidget(hint_label)

        self._static_checkbox: QCheckBox = QCheckBox(
            "Make value static and disable the range"
        )
        layout.addWidget(self._static_checkbox)

        rows: list[tuple[str, QDoubleSpinBox]] = []
        for caption, bound in (("Lower bound", lo), ("Upper bound", hi)):
            row: QHBoxLayout = QHBoxLayout()
            row.setSpacing(6)
            caption_label: QLabel = QLabel(caption)
            spin: QDoubleSpinBox = QDoubleSpinBox()
            spin.setRange(BOUND_RANGE_MIN, BOUND_RANGE_MAX)
            spin.setDecimals(BOUND_DECIMALS)
            spin.setValue(bound)
            row.addWidget(caption_label)
            row.addWidget(spin)
            layout.addLayout(row)
            rows.append((caption, spin))

        self._min_spin: QDoubleSpinBox = rows[0][1]
        self._max_spin: QDoubleSpinBox = rows[1][1]

        buttons: QDialogButtonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        layout.addWidget(buttons)
        self._ok_button: QPushButton = buttons.button(
            QDialogButtonBox.StandardButton.Ok
        )

        # The button box does not wire itself to its parent dialog,
        # so the standard buttons are connected explicitly.
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self._min_spin.valueChanged.connect(self._validate)
        self._max_spin.valueChanged.connect(self._validate)
        self._static_checkbox.toggled.connect(self._on_static_toggled)
        self._validate(0.0)

    def run(self) -> tuple[float, float] | None:
        """
        Show the dialog modally and collect the user's choice.

        Returns:
            The (lower, upper) bounds the user accepted.  When the
            static option is checked the result is degenerate
            (the current value, twice) and pins the parameter;
            otherwise it is the spin boxes' range.  None when the
            dialog was cancelled.
        """
        if self.exec() == QDialog.DialogCode.Accepted:
            if self._static_checkbox.isChecked():
                return (self._value, self._value)
            return (self._min_spin.value(), self._max_spin.value())
        return None

    def _on_static_toggled(self, checked: bool) -> None:
        """
        Freeze both spin boxes at the current value while the
        static option is on; restore the backed-up values when it
        is switched off again.
        """
        spins: list[QDoubleSpinBox] = [self._min_spin, self._max_spin]
        if checked:
            self._backed_up = (
                self._min_spin.value(),
                self._max_spin.value(),
            )
            for spin in spins:
                spin.blockSignals(True)
                spin.setValue(self._value)
                spin.blockSignals(False)
            for spin in spins:
                spin.setEnabled(False)
        else:
            for spin, backed in zip(spins, self._backed_up, strict=True):
                spin.blockSignals(True)
                spin.setValue(backed)
                spin.blockSignals(False)
            for spin in spins:
                spin.setEnabled(True)
        self._validate(0.0)

    def _validate(self, _value: float) -> None:
        """
        Enable OK when the choice is usable: either a strict
        (lower < upper) range, or the static option, which pins the
        parameter and needs no range at all.
        """
        if self._static_checkbox.isChecked():
            self._ok_button.setEnabled(True)
            return
        self._ok_button.setEnabled(
            self._min_spin.value() < self._max_spin.value()
        )


class ThemedGroup(QWidget):
    """
    A collapsible, optionally colored parameter-group container.

    The header is a checkable, auto-raise tool button carrying the
    group's title and a collapse arrow; the content widget holding
    the group's cells is shown or hidden with the button.  A group
    built with a `Theme` gets the theme's light background, an
    accent top border, and an accent-colored bold title; the
    neutral group (the plot selection) is built with `theme=None`
    and keeps the default palette.  The title button is sized to
    fit its text and left-aligned in a header row of the band, so
    every group's title starts at the panel's left edge (a
    stretched button would center the title, because `QToolButton`
    ignores the `text-align` style property).

    The content is a single vertical column by default.  A group
    built with `grid_columns` lays its cells out in a grid with that
    many columns, read left to right, top to bottom: each cell takes
    one column, and a header added with `add_header` starts a new
    row and spans all of the columns.

    In a themed grid group, every cell is drawn as part of a
    visible grid: a thin border and an alternating row shading,
    tinted by the theme (see `GRID_STYLES`), while the full-width
    headers stay undecorated.  A neutral group's cells keep the
    default look.

    Args:
        title: The title shown in the group's header.
        theme: The theme to apply, or None for the neutral group.
        collapsed: Whether the group starts collapsed.
        parent: The parent widget; defaults to None.
        grid_columns: The number of columns for the content grid,
            or None for a single vertical column.
    """

    def __init__(
        self,
        title: str,
        theme: Theme | None,
        collapsed: bool = False,
        parent: QWidget | None = None,
        grid_columns: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._title: str = title
        self._theme: Theme | None = theme
        self._grid_columns: int | None = grid_columns
        self._grid_pos: int = 0
        self._grid_stretch_row: int = 0
        self._grid_positions: dict[int, tuple[int, int, int]] = {}

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        self._toggle: QToolButton = QToolButton(self)
        self._toggle.setCheckable(True)
        self._toggle.setAutoRaise(True)
        # The button is sized to fit its title rather than stretched
        # across the band: `QToolButton` ignores the `text-align`
        # style property, so a stretched button would center the
        # title.  Left-aligning the fitted button in the header row
        # below puts every title at the panel's left edge instead.
        self._toggle.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )

        header_row: QHBoxLayout = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.addWidget(self._toggle)
        header_row.addStretch(1)

        self._content: QWidget = QWidget(self)
        if grid_columns is not None:
            grid: QGridLayout = QGridLayout(self._content)
            grid.setContentsMargins(4, 0, 4, 4)
            grid.setSpacing(2)
            for column_index in range(grid_columns):
                grid.setColumnStretch(column_index, 1)
            grid.setRowStretch(0, 1)
            self._content_layout: QGridLayout | QVBoxLayout = grid
        else:
            column: QVBoxLayout = QVBoxLayout(self._content)
            column.setContentsMargins(4, 0, 4, 4)
            column.addStretch()
            self._content_layout = column

        layout.addLayout(header_row)
        layout.addWidget(self._content)

        # The 4px left padding matches the content's left margin, so
        # a group's title lines up with the cells beneath it, themed
        # or not.
        padding: str = "padding: 4px 0 4px 4px;"
        if theme is not None:
            self.setObjectName(f"param_group_{id(self)}")
            self.setStyleSheet(
                f"#param_group_{id(self)} {{"
                f" background-color: {theme.light};"
                f" border-top: 3px solid {theme.accent};"
                " }"
                "QToolButton {"
                f" color: {theme.accent};"
                f" font-weight: bold; border: none; {padding}"
                " }"
            )
        else:
            self._toggle.setStyleSheet(f"QToolButton {{ {padding} }}")

        self._toggle.setChecked(not collapsed)
        self._set_header_text()
        self._content.setVisible(not collapsed)
        self._toggle.toggled.connect(self._on_toggled)

    @property
    def theme(self) -> Theme | None:
        """The theme applied to this group, or None for a neutral one."""
        return self._theme

    def content(self) -> QWidget:
        """
        The content widget holding the group's cells.

        Returns:
            The content widget, for membership tests when a cell
            must be moved between groups.
        """
        return self._content

    def add_widget(self, widget: QWidget) -> None:
        """
        Add `widget` to the group's content, in the next free slot.

        In a grid group the slot is the next cell in reading order;
        in a column group the widget is inserted above the trailing
        stretch.

        Args:
            widget: The cell or caption to add.
        """
        if isinstance(self._content_layout, QGridLayout):
            assert self._grid_columns is not None
            grid: QGridLayout = self._content_layout
            self._place_widget(grid, widget, 1, self._grid_columns)
        else:
            assert isinstance(self._content_layout, QVBoxLayout)
            column: QVBoxLayout = self._content_layout
            column.insertWidget(column.count() - 1, widget)

    def add_header(self, header: QWidget) -> None:
        """
        Add a full-width `header` to the group's content.

        In a grid group the header starts a new row and spans all
        the columns; in a column group it is added like any other
        widget.

        Args:
            header: The header to add (a zone or mineral label, a
                caption).
        """
        if isinstance(self._content_layout, QGridLayout):
            assert self._grid_columns is not None
            columns: int = self._grid_columns
            grid: QGridLayout = self._content_layout
            if self._grid_pos % columns != 0:
                self._grid_pos += columns - self._grid_pos % columns
            self._place_widget(grid, header, columns, columns)
        else:
            self.add_widget(header)

    def _place_widget(
        self,
        grid: QGridLayout,
        widget: QWidget,
        span: int,
        columns: int,
    ) -> None:
        """
        Place `widget` in the next free grid slot, spanning `span`
        columns, and advance the slot pointer.

        In a themed group, a cell (a widget spanning fewer than all
        the columns) is also drawn with the grid's border and the
        band of its cell row (see `_style_grid_cell`).

        A single stretch row is kept after the last placed row, so
        the content stays pinned to the top of the group.

        Args:
            grid: The content grid to place into.
            widget: The widget to place.
            span: How many columns the widget occupies.
            columns: The grid's total number of columns.
        """
        row: int = self._grid_pos // columns
        col: int = self._grid_pos % columns
        grid.addWidget(widget, row, col, 1, span)
        self._grid_positions[id(widget)] = (row, col, span)
        if span < columns:
            self._style_grid_cell(widget, row)
        self._grid_pos += span
        if self._grid_stretch_row <= row:
            grid.setRowStretch(self._grid_stretch_row, 0)
            self._grid_stretch_row = row + 1
            grid.setRowStretch(self._grid_stretch_row, 1)

    def _cell_row_parity(self, row: int) -> int:
        """
        The alternating band of the cell row `row`: 0 for the
        group's first cell row, 1 for its second, and so on.

        Only rows that hold cells are counted, so the full-width
        zone and mineral headers do not disturb the alternation.

        Args:
            row: The grid row of the cell to look up.

        Returns:
            The row's band: 0 (the first band) or 1 (the second).
        """
        assert self._grid_columns is not None
        columns: int = self._grid_columns
        cell_rows_above: int = len(
            {
                position[0]
                for position in self._grid_positions.values()
                if position[0] < row and position[2] < columns
            }
        )
        return cell_rows_above % 2

    def _style_grid_cell(self, widget: QWidget, row: int) -> None:
        """
        Draw a cell of a themed grid: a 1px border and the band of
        its cell row (see `GRID_STYLES`).

        The rule is scoped to the cell's object name, so it does
        not cascade into the cell's children (the symbol label and
        the spin box keep their own styles).  Cells of a neutral
        group are left undecorated, as are the full-width headers.

        Args:
            widget: The cell to draw.
            row: The grid row the cell occupies.
        """
        if self._theme is None:
            return
        band_a, band_b, line = GRID_STYLES[self._theme.name]
        band: str = band_a if self._cell_row_parity(row) == 0 else band_b
        object_name: str = f"grid_cell_{id(widget)}"
        widget.setObjectName(object_name)
        widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        widget.setStyleSheet(
            f"#{object_name} {{"
            f" background-color: {band};"
            f" border: 1px solid {line};"
            " }"
        )

    def replace_widget(self, old: QWidget, new: QWidget) -> None:
        """
        Replace `old` with `new` at the same position in the content.

        The old widget is removed from the layout and scheduled for
        deletion.  In a themed grid group, a replacement cell is
        re-drawn with the grid's border and the band of the row it
        takes over, so an unlocked parameter joins the grid like
        any other cell.

        Args:
            old: The widget to replace (a member of the content).
            new: The widget that takes its place.

        Raises:
            ValueError: Raised when `old` is not in the content
                layout.
        """
        if isinstance(self._content_layout, QGridLayout):
            assert self._grid_columns is not None
            columns: int = self._grid_columns
            grid: QGridLayout = self._content_layout
            position: tuple[int, int, int] | None = (
                self._grid_positions.pop(id(old), None)
            )
            if position is None:
                raise ValueError(
                    "Cannot replace a widget that is not in this "
                    "group's content."
                )
            row, col, span = position
            grid.removeWidget(old)
            old.deleteLater()
            grid.addWidget(new, row, col, 1, span)
            self._grid_positions[id(new)] = (row, col, span)
            if span < columns:
                self._style_grid_cell(new, row)
        else:
            assert isinstance(self._content_layout, QVBoxLayout)
            column: QVBoxLayout = self._content_layout
            index: int = column.indexOf(old)
            if index < 0:
                raise ValueError(
                    "Cannot replace a widget that is not in this "
                    "group's content."
                )
            column.removeWidget(old)
            old.deleteLater()
            column.insertWidget(index, new)

    def _set_header_text(self) -> None:
        """Restate the header with the arrow of the current state."""
        arrow: str = (
            ARROW_EXPANDED if self._toggle.isChecked() else ARROW_COLLAPSED
        )
        self._toggle.setText(f"{arrow} {self._title}")

    def _on_toggled(self, checked: bool) -> None:
        """Show or hide the content with the toggle button's state."""
        self._content.setVisible(checked)
        self._set_header_text()
    # =================== #


# ==== Public ==== #
class ParamPanel(QScrollArea):
    """
    The parameter panel of the main window: every adjustable model
    parameter, grouped by zone below the figure, plus the
    figure-selection controls.

    The groups, in order, are: *Hydrology* (blue-themed; one cell
    per zone parameter, in the spec's zone order), *Catchment
    proportion* (the single area-fraction parameter, only when the
    model has one), *Reactive transport* (red-themed; per-zone
    storage parameters and per-mineral kinetic parameters in every
    reactive zone), *Initial state* (red-themed; the calibrated
    exchange-site state per speciation zone), *Plots* (the
    streamflow toggle and river-species checkboxes of the
    river-concentration figure, the zone checkboxes of the storage
    and zone-concentration figures, the single mineral of the
    reaction-rate figure, and the two date pickers of the
    plotting window), and *History* (the previous-runs dropdown, the
    *Overlay previous results* checkbox, and the *Use these
    parameters* button).  The history group keeps no parameter
    cells, so a bounds reset leaves it in place, and the window
    fills its dropdown and answers its two controls.

    In hydrology-only mode (the main window's *Hydrology only*
    checkbox), the *Reactive transport* and *Initial state* groups
    and the streamflow, river-species, and mineral controls of the
    *Plots* group are hidden by `set_hydro_only`; the zone checkboxes
    and the date window stay visible.  The hidden controls keep
    their state, so `selection()` still reports it (the window needs
    it when the user switches back to full mode).

    The four parameter groups lay their cells out in a five-column
    grid: each cell (the symbol to the left of one parameter's
    value) takes one column, so a row holds five, the most
    parameters any zone carries, and a zone or mineral header spans
    all of them.  Every group's title is left-aligned, so the
    titles line up at the panel's left edge.  The *Plots* group's
    controls take two per row.  In a themed group the cells are
    drawn as a visible grid (a thin border and an alternating row
    shading; see `ThemedGroup`).  Free parameters (distinct
    bounds) are editable spin-box cells; degenerate parameters
    (identical bounds) are pinned and shown as read-only cells that
    unlock on double-click.  A free cell's range can be re-edited
    by double-clicking the cell.

    Signals:
        value_changed: Emitted with the parameter key whenever a
            value changes (a spin box was edited, a range was
            edited, or a saved state was restored).
        bounds_changed: Emitted with the parameter key and its new
            (lower, upper) bounds when a range is edited or a pinned
            parameter is unlocked.
        selection_changed: Emitted with the current `PlotSelection`
            whenever a plot control changes.
        history_changed: Emitted (with no payload) whenever the
            history dropdown's selection or the overlay checkbox
            changes; the window then rebuilds the figures from the
            last results, re-applying or dropping the overlay.
        history_restore_requested: Emitted when the *Use these
            parameters* button is clicked; the window restores the
            selected run's parameters without a re-run.

    Args:
        input_data: The loaded calibration inputs (parameter bounds
            and the reaction network).
        selection: The plot selection to start from.
        parent: The parent widget; defaults to None.
    """

    value_changed = Signal(str)
    bounds_changed = Signal(str, float, float)
    selection_changed = Signal(PlotSelection)
    history_changed = Signal()
    history_restore_requested = Signal()

    def __init__(
        self,
        input_data: GuiInputData,
        selection: PlotSelection,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._input_data: GuiInputData = input_data
        self._bounds: dict[str, tuple[float, float]] = dict(input_data.bounds)
        # The default of a free parameter is the midpoint of its
        # bounds; a degenerate parameter is pinned to its bound.
        self._values: dict[str, float] = {
            key: (lo if lo == hi else 0.5 * (lo + hi))
            for key, (lo, hi) in input_data.bounds.items()
        }
        self._rows: dict[str, FreeCell | PinnedCell] = {}
        self._groups: dict[str, ThemedGroup] = {}
        # Hydrology-only mode: the reactive-transport groups and
        # checkboxes are hidden (`set_hydro_only`).
        self._hydro_only: bool = False

        container: QWidget = QWidget()
        self._container_layout: QVBoxLayout = QVBoxLayout(container)
        self._container_layout.setContentsMargins(8, 8, 8, 8)
        self._container_layout.setSpacing(0)
        self.setWidget(container)

        self._build_param_groups(first_build=True)
        self._plots_group: ThemedGroup = self._build_plots_group(selection)
        self._container_layout.addWidget(self._plots_group)
        self._history_group: ThemedGroup = self._build_history_group()
        self._container_layout.addWidget(self._history_group)
        self._container_layout.addStretch()

    def param_values(self) -> dict[str, float]:
        """
        The current value of every parameter.

        Returns:
            A mapping of parameter key to value, in the bounds' key
            order.
        """
        return {key: self._values[key] for key in self._input_data.bounds}

    def current_bounds(self) -> dict[str, tuple[float, float]]:
        """
        The panel's current bounds for every parameter.

        The bounds are the loaded defaults until the user edits them
        by double-clicking a free cell (or unlocks a pinned
        parameter); `reset_bounds` restores the defaults.

        Returns:
            A mapping of parameter key to (lower, upper) bound, in
            the bounds' key order.
        """
        return {key: self._bounds[key] for key in self._input_data.bounds}

    def selection(self) -> PlotSelection:
        """
        The current figure selection, as held by the controls' state.

        Returns:
            The `PlotSelection` of the checked river species (in the
            panel's order), the checked zones (in the panel's zone
            order), the selected mineral, the streamflow toggle, and
            the plotting window's dates.
        """
        return PlotSelection(
            species=[
                name
                for name, box in self._species_boxes.items()
                if box.isChecked()
            ],
            zones=[
                zone
                for zone, box in self._zone_boxes.items()
                if box.isChecked()
            ],
            mineral=self._mineral_combo.currentText(),
            streamflow=self._streamflow_box.isChecked(),
            window_start=self._window_from_edit.date().toString(
                Qt.DateFormat.ISODate
            ),
            window_end=self._window_to_edit.date().toString(
                Qt.DateFormat.ISODate
            ),
        )

    def set_selection(self, selection: PlotSelection) -> None:
        """
        Set the controls' state to `selection`.

        The controls' signals are blocked while this runs, so the
        panel does not re-emit `selection_changed`.  Names not in the
        panel's controls (a selection saved against a different
        network, or species that have since been removed) are
        ignored, and the plotting window is clamped so its start
        does not exceed its end.

        Args:
            selection: The plot selection to apply.
        """
        self._streamflow_box.blockSignals(True)
        self._streamflow_box.setChecked(selection.streamflow)
        self._streamflow_box.blockSignals(False)
        for name, box in self._species_boxes.items():
            box.blockSignals(True)
            box.setChecked(name in selection.species)
            box.blockSignals(False)
        for zone, box in self._zone_boxes.items():
            box.blockSignals(True)
            box.setChecked(zone in selection.zones)
            box.blockSignals(False)
        # The combo keeps its current entry when the saved mineral
        # is not among its items.
        self._mineral_combo.blockSignals(True)
        mineral_index: int = self._mineral_combo.findText(selection.mineral)
        if mineral_index >= 0:
            self._mineral_combo.setCurrentIndex(mineral_index)
        self._mineral_combo.blockSignals(False)
        for edit, value in (
            (self._window_from_edit, selection.window_start),
            (self._window_to_edit, selection.window_end),
        ):
            # A date the picker cannot represent (a selection saved
            # against a different record) is ignored.
            date: QDate = QDate.fromString(value, Qt.DateFormat.ISODate)
            if date.isValid():
                edit.blockSignals(True)
                edit.setDate(date)
                edit.blockSignals(False)
        self._clamp_window()

    def set_hydro_only(self, enabled: bool) -> None:
        """
        Hide (or show) the reactive-transport parts of the panel.

        In hydrology-only mode the *Reactive transport* and
        *Initial state* groups and the streamflow, river-species,
        and mineral controls of the *Plots* group are hidden; the
        zone checkboxes and the plotting window stay visible (the
        hydrology figures are drawn from them too).  The hidden
        controls keep their state, so `selection()` still reports it
        (the window needs it when the user switches back to full
        mode).

        Args:
            enabled: True for hydrology-only mode.
        """
        self._hydro_only = enabled
        for name in ("reactive", "initial"):
            group: ThemedGroup | None = self._groups.get(name)
            if group is not None:
                group.setVisible(not enabled)
        for header in (
            self._streamflow_header,
            self._species_header,
            self._mineral_header,
        ):
            header.setVisible(not enabled)
        self._streamflow_box.setVisible(not enabled)
        for box in self._species_boxes.values():
            box.setVisible(not enabled)
        self._mineral_combo.setVisible(not enabled)

    def reset_bounds(self) -> dict[str, float]:
        """
        Restore the bounds to the loaded defaults and rebuild the
        parameter groups.

        The current values are clamped into the restored ranges and
        the four parameter groups are rebuilt in place (the
        plot-selection group is kept).  A previously unlocked
        parameter is pinned again when its default bounds are
        degenerate.

        Returns:
            The clamped parameter values, keyed by bounds key in the
            bounds' key order.
        """
        self._bounds = dict(self._input_data.bounds)
        for key in self._values:
            lo, hi = self._bounds[key]
            self._values[key] = min(max(self._values[key], lo), hi)
        self._build_param_groups(first_build=False)
        return self.param_values()

    def reload(
        self,
        input_data: GuiInputData,
        selection: PlotSelection,
    ) -> None:
        """
        Replace the loaded inputs and rebuild every panel group.

        Used when the window switches the selected model: the new
        spec's bounds, zones, labels, colors, and chemistry replace
        the old ones, and every parameter starts at its new default
        (a pinned parameter at its bound, a free one at the midpoint
        of its range).  The plot-selection group is rebuilt from
        ``selection`` and the history group is kept (the window
        refills its dropdown).  The hydrology-only visibility is
        re-applied.

        Args:
            input_data: The new model's assembled inputs.
            selection: The plot selection to start from.
        """
        self._input_data = input_data
        self._bounds = dict(input_data.bounds)
        self._values = {
            key: (lo if lo == hi else 0.5 * (lo + hi))
            for key, (lo, hi) in input_data.bounds.items()
        }
        self._rows = {}
        self._groups = {}
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._build_param_groups(first_build=True)
        self._plots_group = self._build_plots_group(selection)
        self._container_layout.addWidget(self._plots_group)
        self._history_group = self._build_history_group()
        self._container_layout.addWidget(self._history_group)
        self._container_layout.addStretch()
        self.set_hydro_only(self._hydro_only)

    def set_values(self, values: dict[str, float]) -> None:
        """
        Update the panel's parameter values in place, keeping the bounds.

        Unlike `restore_state`, the bounds are left exactly as they
        are and the groups are not rebuilt: each value is clamped
        into the current bounds, recorded, and pushed into its free
        cell's spin box.  A pinned cell (degenerate bounds) needs no
        touch — its value is its bound, and the clamped value the
        panel records equals it.  Skipping the rebuild keeps the
        scroll area's position and the cells themselves, so adopting
        a line search's best point leaves the panel where the user
        left it.  No signals are emitted: this is a state set, not a
        user change.

        Args:
            values: The new parameter values, keyed by bounds key.
                Names not in the current bounds are ignored.
        """
        for key, value in values.items():
            if key not in self._bounds:
                continue
            lo, hi = self._bounds[key]
            self._values[key] = min(max(float(value), lo), hi)
            cell: FreeCell | PinnedCell | None = self._rows.get(key)
            if cell is not None and isinstance(cell, FreeCell):
                cell.set_value(self._values[key])

    def restore_state(
        self,
        values: dict[str, float],
        bounds: dict[str, tuple[float, float]],
    ) -> None:
        """
        Apply saved parameter values and bounds to the panel.

        Used at startup to restore a saved run: the saved bounds
        replace the panel's current ones (a parameter whose saved
        bounds are degenerate is pinned again, one with a range is a
        free cell again), and the saved values are clamped into the
        saved bounds.  Names not in the loaded bounds (a state saved
        against a different network) are ignored; names missing from
        the saved values fall back to their default (the midpoint of
        the bounds).  No signals are emitted: this is a state set,
        not a user change.

        Args:
            values: The saved parameter values, keyed by bounds key.
            bounds: The saved (lower, upper) bounds, keyed by bounds
                key.
        """
        self._bounds = {
            key: (float(lo), float(hi))
            for key, (lo, hi) in bounds.items()
            if key in self._input_data.bounds
        }
        self._values = {}
        for key, (lo, hi) in self._bounds.items():
            if key in values:
                value: float = min(max(float(values[key]), lo), hi)
            else:
                value = lo if lo == hi else 0.5 * (lo + hi)
            self._values[key] = value
        self._build_param_groups(first_build=False)

    # =================== #

    def _build_param_groups(self, first_build: bool) -> None:
        """
        Build (or rebuild, after `reset_bounds`) the parameter
        groups from the current bounds.

        A group the selected model does not have (the catchment
        proportion of a single-column model) is skipped.  A rebuild
        creates fresh (visible) groups, so the hydrology-only mode's
        visibility is re-applied at the end.

        Args:
            first_build: True for the initial build (the groups are
                appended to an empty container); False for a rebuild
                (the old groups are removed and the new ones inserted
                before the plot-selection group).
        """
        if first_build:
            insert_at: int = 0
        else:
            for group in self._groups.values():
                self._container_layout.removeWidget(group)
                group.deleteLater()
            insert_at = self._container_layout.indexOf(self._plots_group)

        candidates: dict[str, ThemedGroup | None] = {
            "hydrology": self._populate_hydrology(),
            "proportion": self._populate_proportion(),
            "reactive": self._populate_reactive(),
            "initial": self._populate_initial(),
        }
        self._groups = {
            name: group
            for name, group in candidates.items()
            if group is not None
        }
        for offset, group in enumerate(self._groups.values()):
            self._container_layout.insertWidget(insert_at + offset, group)
        if self._hydro_only:
            for name in ("reactive", "initial"):
                optional_group: ThemedGroup | None = self._groups.get(name)
                if optional_group is not None:
                    optional_group.setVisible(False)

    def _populate_hydrology(self) -> ThemedGroup:
        """The blue-themed hydrology group: one cell per zone parameter."""
        group: ThemedGroup = ThemedGroup(
            "Hydrology", HYDRO_THEME, grid_columns=GRID_COLUMNS
        )
        spec = self._input_data.spec
        for zone in spec.zones:
            group.add_header(
                ZoneHeader(
                    zone, spec.zone_labels[zone], spec.zone_colors[zone]
                )
            )
            zone_class: type[pt.HydrologicZone] = (
                spec.model_class.get_zone_type(zone)
            )
            for param in zone_class.parameter_names():
                self._add_param_cell(group, f"{zone}.{param}")
        return group

    def _populate_proportion(self) -> ThemedGroup | None:
        """
        The blue-themed catchment-proportion group: one cell, or None
        when the model has no catchment split.
        """
        if "proportion.1" not in self._bounds:
            return None
        group: ThemedGroup = ThemedGroup(
            "Catchment proportion", HYDRO_THEME, grid_columns=GRID_COLUMNS
        )
        self._add_param_cell(group, "proportion.1")
        caption: QLabel = QLabel(
            "Fraction of the catchment area in the distal (hillslope) "
            "column; the riparian column receives the remaining fraction."
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("color: gray;")
        caption.setContentsMargins(4, 2, 4, 0)
        group.add_header(caption)
        return group

    def _populate_reactive(self) -> ThemedGroup:
        """
        The red-themed reactive-transport group: per-zone storage
        parameters, and per-mineral kinetic parameters in every
        reactive zone.
        """
        group: ThemedGroup = ThemedGroup(
            "Reactive transport", RT_THEME, grid_columns=GRID_COLUMNS
        )
        spec = self._input_data.spec
        mineral_names: list[str] = self._input_data.network.mineral_names
        for zone in spec.zones:
            group.add_header(
                ZoneHeader(
                    zone, spec.zone_labels[zone], spec.zone_colors[zone]
                )
            )
            for param in RT_DIM_PARAMS:
                self._add_param_cell(group, f"{zone}.{param}")
            if spec.zone_configs[zone].do_reactions:
                for mineral_id, mineral_name in enumerate(mineral_names):
                    group.add_header(BoldHeader(mineral_name, top_margin=4))
                    for field in MINERAL_PARAMS:
                        self._add_param_cell(
                            group, f"{zone}.{field}[{mineral_id}]"
                        )
        return group

    def _populate_initial(self) -> ThemedGroup | None:
        """
        The red-themed initial-state group: the calibrated
        exchange-site state per speciation zone.

        None when the network has no exchange site: the bounds
        carry no initial-state cells then (the gate `data`
        applies when it builds them), so the group has nothing
        to show.
        """
        if not self._input_data.network.has_exchange:
            return None
        group: ThemedGroup = ThemedGroup(
            "Initial state", RT_THEME, grid_columns=GRID_COLUMNS
        )
        for zone in self._input_data.spec.speciation_zones:
            self._add_param_cell(group, f"initial_state.{zone}_x")
        return group

    def _add_param_cell(self, group: ThemedGroup, key: str) -> None:
        """
        Add the cell for one parameter to `group`, as a free spin-box
        cell or a pinned cell according to its current bounds.
        """
        symbol: str = param_symbol(key)
        label: str = param_label(
            key,
            self._input_data.network.mineral_names,
            zone_labels=self._input_data.spec.zone_labels,
        )
        lo, hi = self._bounds[key]
        if lo == hi:
            pinned_cell: PinnedCell = PinnedCell(
                key, symbol, label, self._values[key]
            )
            pinned_cell.unlock_requested.connect(self._on_unlock_requested)
            self._rows[key] = pinned_cell
        else:
            free_cell: FreeCell = FreeCell(
                key, symbol, label, (lo, hi), self._values[key], group.theme
            )
            free_cell.value_changed.connect(self._on_value_changed)
            free_cell.edit_requested.connect(self._on_bounds_edit)
            self._rows[key] = free_cell
        group.add_widget(self._rows[key])

    def _build_plots_group(self, selection: PlotSelection) -> ThemedGroup:
        """
        The neutral "Plots" group: the controls that select what the
        figures draw, in figure order — the river-concentration
        figure's streamflow toggle, the river-species checkboxes
        (shared with the zone-concentration figure), the zone
        checkboxes (shared with the storage figure), the
        reaction-rate figure's single mineral, and the two date
        pickers that set the plotting window of every figure.

        The streamflow, species, and mineral headers are hidden
        together with their controls in hydrology-only mode, so
        they are kept for `set_hydro_only`; the zone and window
        sections stay visible in both modes.

        Args:
            selection: The control state to start from.
        """
        group: ThemedGroup = ThemedGroup("Plots", None, grid_columns=2)

        self._streamflow_header: QWidget = BoldHeader(
            "Streamflow (river-concentration figure)"
        )
        group.add_header(self._streamflow_header)
        self._streamflow_box: QCheckBox = QCheckBox(
            "Simulated and measured streamflow (inverted axis)"
        )
        self._streamflow_box.toggled.connect(self._on_selection_changed)
        group.add_widget(self._streamflow_box)

        self._species_header: QWidget = BoldHeader(
            "River species (river- and zone-concentration figures)"
        )
        group.add_header(self._species_header)
        self._species_boxes: dict[str, QCheckBox] = {}
        for name in self._input_data.conc_df.columns:
            box: QCheckBox = QCheckBox(name)
            box.toggled.connect(self._on_selection_changed)
            self._species_boxes[name] = box
            group.add_widget(box)

        self._zone_header: QWidget = BoldHeader(
            "Zones (storage and zone-concentration figures)"
        )
        group.add_header(self._zone_header)
        self._zone_boxes: dict[str, QCheckBox] = {}
        spec = self._input_data.spec
        for zone in spec.zones:
            box = QCheckBox(spec.zone_labels[zone])
            box.toggled.connect(self._on_selection_changed)
            self._zone_boxes[zone] = box
            group.add_widget(box)

        self._mineral_header: QWidget = BoldHeader(
            "Mineral (reaction-rate figure)"
        )
        group.add_header(self._mineral_header)
        self._mineral_combo: QComboBox = QComboBox()
        for name in self._input_data.network.mineral_names:
            self._mineral_combo.addItem(name)
        self._mineral_combo.currentIndexChanged.connect(
            self._on_mineral_changed
        )
        group.add_widget(self._mineral_combo)

        self._window_header: QWidget = BoldHeader(
            "Plotting period (all figures, dates)"
        )
        group.add_header(self._window_header)
        # The pickers span the loaded record's window, so a date
        # outside the record cannot be selected.
        window_start: str = self._input_data.window[0].strftime("%Y-%m-%d")
        window_end: str = self._input_data.window[1].strftime("%Y-%m-%d")
        self._window_from_row: QWidget
        self._window_from_edit: QDateEdit
        self._window_from_row, self._window_from_edit = labeled_date_edit(
            "From",
            DEFAULT_WINDOW_START,
            window_start,
            window_end,
        )
        self._window_from_edit.dateChanged.connect(
            self._on_window_from_changed
        )
        group.add_widget(self._window_from_row)
        self._window_to_row: QWidget
        self._window_to_edit: QDateEdit
        self._window_to_row, self._window_to_edit = labeled_date_edit(
            "To",
            DEFAULT_WINDOW_END,
            window_start,
            window_end,
        )
        self._window_to_edit.dateChanged.connect(
            self._on_window_to_changed
        )
        group.add_widget(self._window_to_row)

        self.set_selection(selection)
        return group

    def _build_history_group(self) -> ThemedGroup:
        """
        The neutral "History" group: the previous-runs dropdown, the
        *Overlay previous results* checkbox, and the *Use these
        parameters* button.

        The group keeps no parameter cells, so it stays out of
        `self._groups` and a bounds reset leaves it in place; the
        window fills the dropdown (with the saved run directories
        and the session's runs, newest first) and answers the two
        controls.
        """
        group: ThemedGroup = ThemedGroup("History", None)

        self._history_combo: QComboBox = QComboBox()
        self._history_combo.setToolTip(
            "The previous runs: the saved run directories and this "
            "session's runs, newest first."
        )
        self._history_combo.currentIndexChanged.connect(
            self._on_history_changed
        )
        group.add_widget(self._history_combo)

        self._history_overlay_checkbox: QCheckBox = QCheckBox(
            "Overlay previous results"
        )
        self._history_overlay_checkbox.setToolTip(
            "Draw the selected previous run's results faintly (half "
            "transparent) on the current figures."
        )
        self._history_overlay_checkbox.toggled.connect(
            self._on_history_changed
        )
        group.add_widget(self._history_overlay_checkbox)

        restore_button: QPushButton = QPushButton("Use these parameters")
        restore_button.setToolTip(
            "Restore the selected previous run's parameters to the "
            "panel, without a re-run."
        )
        restore_button.clicked.connect(self._on_history_restore)
        group.add_widget(restore_button)

        return group

    def set_history_items(self, labels: list[str]) -> None:
        """
        Replace the history dropdown's items.

        The dropdown's signals are blocked while the items are
        replaced, so the change does not re-emit
        `history_changed`; the previous selection is kept when its
        label is still present.

        Args:
            labels: The labels of the previous runs, newest first.
        """
        previous: str = self._history_combo.currentText()
        self._history_combo.blockSignals(True)
        self._history_combo.clear()
        self._history_combo.addItems(labels)
        index: int = self._history_combo.findText(previous)
        if index >= 0:
            self._history_combo.setCurrentIndex(index)
        self._history_combo.blockSignals(False)

    def history_selected_index(self) -> int:
        """
        The index of the dropdown's selected previous run.

        Returns:
            The index, or -1 when the dropdown is empty.
        """
        return self._history_combo.currentIndex()

    def history_overlay_enabled(self) -> bool:
        """Whether the *Overlay previous results* checkbox is on."""
        return self._history_overlay_checkbox.isChecked()

    def _on_value_changed(self, key: str) -> None:
        """Record a free cell's new value and re-emit it to the window."""
        cell: FreeCell | PinnedCell = self._rows[key]
        if isinstance(cell, FreeCell):
            self._values[key] = cell.value()
        self.value_changed.emit(key)

    def _on_bounds_edit(self, key: str) -> None:
        """
        Open the range dialog for `key` and apply the user's choice:
        a strict range replaces the cell's bounds, while a degenerate
        (static) one pins the parameter at its displayed value.
        """
        label: str = param_label(
            key,
            self._input_data.network.mineral_names,
            zone_labels=self._input_data.spec.zone_labels,
        )
        lo, hi = self._bounds[key]
        decimals: int = bound_decimals(lo, hi)
        hint: str = f"Current range: {lo:.{decimals}f} to {hi:.{decimals}f}."
        result: tuple[float, float] | None = BoundsDialog(
            f"Edit range — {label}", lo, hi, hint, self._values[key], self
        ).run()
        if result is None:
            return
        new_lo, new_hi = result

        cell: FreeCell | PinnedCell = self._rows[key]
        if not isinstance(cell, FreeCell):
            return
        if new_lo == new_hi:
            # Static value: pin the parameter at the displayed value.
            v: float = new_lo
            self._bounds[key] = (v, v)
            self._values[key] = v
            # Convert exactly as the group builder does.
            group: ThemedGroup = self._group_of(key)
            pinned: PinnedCell = PinnedCell(
                key, param_symbol(key), label, v
            )
            group.replace_widget(cell, pinned)
            pinned.unlock_requested.connect(self._on_unlock_requested)
            self._rows[key] = pinned
            self.bounds_changed.emit(key, v, v)
            self.value_changed.emit(key)
        else:
            self._bounds[key] = (new_lo, new_hi)
            cell.set_bounds(new_lo, new_hi)
            self._values[key] = cell.value()
            self.bounds_changed.emit(key, new_lo, new_hi)
            self.value_changed.emit(key)

    def _on_unlock_requested(self, key: str) -> None:
        """
        Open the range dialog for a pinned parameter and, if the user
        accepts a strict range, convert the cell into a free cell in
        place.  A degenerate (static) result is a no-op: the cell is
        already pinned at that value.
        """
        label: str = param_label(
            key,
            self._input_data.network.mineral_names,
            zone_labels=self._input_data.spec.zone_labels,
        )
        value: float = self._values[key]
        # A sensible first range around the pinned value: a symmetric
        # span of half the value, or ±1 for a value pinned at zero.
        span: float = abs(value) / 2 if value != 0 else 1.0
        lo, hi = value - span, value + span
        hint: str = (
            f"{label} is pinned to {value:.4f}.  Choose a range to make "
            "it adjustable; the current value is kept."
        )
        result: tuple[float, float] | None = BoundsDialog(
            f"Edit range — {label}", lo, hi, hint, value, self
        ).run()
        if result is None:
            return
        new_lo, new_hi = result
        if new_lo == new_hi:
            return
        self._bounds[key] = (new_lo, new_hi)
        self._values[key] = min(max(self._values[key], new_lo), new_hi)

        group: ThemedGroup = self._group_of(key)
        free_cell: FreeCell = FreeCell(
            key,
            param_symbol(key),
            label,
            (new_lo, new_hi),
            self._values[key],
            group.theme,
        )
        free_cell.value_changed.connect(self._on_value_changed)
        free_cell.edit_requested.connect(self._on_bounds_edit)
        group.replace_widget(self._rows[key], free_cell)
        self._rows[key] = free_cell

        self.bounds_changed.emit(key, new_lo, new_hi)
        self.value_changed.emit(key)

    def _on_selection_changed(self, _checked: bool) -> None:
        """A plot checkbox was toggled: recompute and emit the selection."""
        self.selection_changed.emit(self.selection())

    def _on_mineral_changed(self, _index: int) -> None:
        """The mineral combo moved: recompute and emit the selection."""
        self.selection_changed.emit(self.selection())

    def _on_window_from_changed(self, _date: QDate) -> None:
        """
        The *from* date moved: lift the *to* date to at least the
        *from* date, then emit the new window.
        """
        if self._window_to_edit.date() < self._window_from_edit.date():
            self._window_to_edit.blockSignals(True)
            self._window_to_edit.setDate(self._window_from_edit.date())
            self._window_to_edit.blockSignals(False)
        self.selection_changed.emit(self.selection())

    def _on_window_to_changed(self, _date: QDate) -> None:
        """
        The *to* date moved: lower the *from* date to at most the
        *to* date, then emit the new window.
        """
        if self._window_from_edit.date() > self._window_to_edit.date():
            self._window_from_edit.blockSignals(True)
            self._window_from_edit.setDate(self._window_to_edit.date())
            self._window_from_edit.blockSignals(False)
        self.selection_changed.emit(self.selection())

    def _clamp_window(self) -> None:
        """
        Keep the plotting window ordered without emitting: when the
        *from* date stands past the *to* date, the *to* date is
        lifted to it.  Used by `set_selection`, whose signal blocks
        make the emitted change redundant.
        """
        if self._window_to_edit.date() < self._window_from_edit.date():
            self._window_to_edit.blockSignals(True)
            self._window_to_edit.setDate(self._window_from_edit.date())
            self._window_to_edit.blockSignals(False)

    def _on_history_changed(self) -> None:
        """Re-emit a change of the history dropdown or the overlay checkbox."""
        self.history_changed.emit()

    def _on_history_restore(self) -> None:
        """Re-emit the *Use these parameters* button's click."""
        self.history_restore_requested.emit()

    def _group_of(self, key: str) -> ThemedGroup:
        """
        The parameter group containing the cell for `key`.

        Args:
            key: The parameter's bounds key.

        Returns:
            The group whose content widget is the cell's parent.

        Raises:
            KeyError: Raised when no group contains the cell.
        """
        cell: FreeCell | PinnedCell = self._rows[key]
        for group in self._groups.values():
            if cell.parentWidget() is group.content():
                return group
        raise KeyError(f"No parameter group found for {key!r}.")
    # =================== #
# =============== #
