"""
The parameter panel of the East River lateral-model calibration GUI.

Holds the model's adjustable parameters in the panel below the
figure: the hydrology parameters of every zone (blue-themed), the
single catchment-proportion parameter, the reactive-transport
parameters of every zone (red-themed, with a header per mineral in
the non-snow zones), and the calibrated initial state.

Every parameter whose bounds are distinct is a spin-box cell whose
range can be re-edited by double-clicking the cell; the value's
displayed precision follows the current range (see `bound_decimals`).
A parameter whose bounds are identical is pinned and shown as a
read-only cell; double-clicking the cell opens a dialog that gives
it a range, converting it into a free cell.

The last group, "Plots", carries the checkboxes that select the
species, zones, and minerals drawn in the zone and reaction-rate
figures, and emits `selection_changed` whenever one of them is
toggled.
"""

from typing import Final

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
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

from potions.gui.common import (
    HYDRO_PARAMS,
    HYDRO_THEME,
    MINERAL_PARAMS,
    RT_DIM_PARAMS,
    RT_THEME,
    SNOW_ZONES,
    SPECIATION_ZONES,
    Theme,
    ZONE_COLORS,
    ZONE_LABELS,
    ZONE_ORDER,
    bound_decimals,
    param_label,
    param_symbol,
)
from potions.gui.data import GuiInputData, PlotSelection

# ==== Constants ==== #
# The arrows of a group's collapse button, expanded or not.
_ARROW_EXPANDED: Final[str] = "▾"
_ARROW_COLLAPSED: Final[str] = "▸"

# The color chip left of a zone subheader, tinted with the zone's
# color from the figure palette.
_ZONE_CHIP: Final[str] = "■"

# The range of the bound-edit dialog's spin boxes: wide enough for
# any of the model's bounds (a few thousands at most), with the
# decimals of the saved parameters.csv files.
_BOUND_RANGE_MIN: Final[float] = -1e6
_BOUND_RANGE_MAX: Final[float] = 1e6
_BOUND_DECIMALS: Final[int] = 6

# The number of columns in the parameter groups' cell grids: the
# symbol and the spin box of each cell, two cells per row.
_GRID_COLUMNS: Final[int] = 4
# =================== #


# ==== Widgets ==== #
class _FreeCell(QWidget):
    """
    A single free-parameter cell: the parameter's symbol and a spin
    box holding its value.

    The spin box is clamped to the cell's (possibly user-edited)
    range, and its decimal precision and step width follow the range
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

        layout: QHBoxLayout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(4)

        self._symbol: QLabel = QLabel(symbol)
        self._symbol.setToolTip(label)
        self._symbol.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._symbol.setMinimumWidth(70)
        if theme is not None:
            self._symbol.setStyleSheet(f"color: {theme.accent};")
        layout.addWidget(self._symbol, 1)

        self._spin: QDoubleSpinBox = QDoubleSpinBox()
        self._spin.setToolTip(label)
        self._spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._spin.valueChanged.connect(self._on_value_edited)
        layout.addWidget(self._spin, 1)

        self._line_edit: QLineEdit | None = self._spin.lineEdit()
        if self._line_edit is not None:
            self._line_edit.installEventFilter(self)

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
        Open the range dialog when the spin box's line edit is
        double-clicked (a click there never reaches the cell's
        double-click handler).

        Args:
            watched: The object the event was delivered to.
            event: The event itself.

        Returns:
            Always False: the event keeps propagating.
        """
        if (
            watched is self._line_edit
            and event.type() == QEvent.Type.MouseButtonDblClick
        ):
            self.edit_requested.emit(self._key)
        return False

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """
        Open the range dialog when any part of the cell is
        double-clicked, then forward the event.
        """
        self.edit_requested.emit(self._key)
        super().mouseDoubleClickEvent(event)


class _PinnedCell(QWidget):
    """
    A read-only cell for a pinned parameter, whose bounds are
    identical so there is nothing to adjust.

    The value is shown in a disabled (grayed) spin box, and the
    symbol carries the parameter's full label as its tooltip.
    Double-clicking the cell emits `unlock_requested`, which the
    panel answers by opening the range dialog and converting the
    cell into a free cell.

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

        layout: QHBoxLayout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(4)

        self._symbol: QLabel = QLabel(symbol)
        self._symbol.setToolTip(label)
        self._symbol.setStyleSheet("color: gray;")
        self._symbol.setMinimumWidth(70)
        layout.addWidget(self._symbol, 1)

        self._spin: QDoubleSpinBox = QDoubleSpinBox()
        self._spin.setRange(_BOUND_RANGE_MIN, _BOUND_RANGE_MAX)
        self._spin.setDecimals(bound_decimals(value, value))
        self._spin.setValue(value)
        self._spin.setEnabled(False)
        layout.addWidget(self._spin, 1)

        self.setToolTip(
            "Pinned to a single value. Double-click to give this "
            "parameter a range."
        )

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """
        Emit `unlock_requested` for this cell, then forward the
        event (a double-click on the disabled spin box is forwarded
        here by Qt).
        """
        self.unlock_requested.emit(self._key)
        super().mouseDoubleClickEvent(event)



class _BoldHeader(QLabel):
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
        top_margin: int = 4,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setText(f"<b>{text}</b>")
        self.setContentsMargins(8, top_margin, 8, 0)


class _ZoneHeader(QLabel):
    """
    The bold subheader of a zone's parameters: the zone's color chip
    (the zone's color from the figure palette) followed by its
    display name.

    Args:
        zone: The zone, a key of `ZONE_LABELS` and `ZONE_COLORS`.
        parent: The parent widget; defaults to None.
    """

    def __init__(self, zone: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(
            f'<span style="color: {ZONE_COLORS[zone]};">{_ZONE_CHIP}</span> '
            f"<b>{ZONE_LABELS[zone]}</b>"
        )
        self.setContentsMargins(0, 8, 0, 0)


class _BoundsDialog(QDialog):
    """
    The range-edit dialog: two spin boxes prefilled with the
    current lower and upper bounds, an optional hint line, and OK /
    Cancel buttons.

    OK stays disabled until the two bounds form a strict range
    (lower < upper); the check runs on every change of either spin
    box.

    Args:
        title: The dialog's window title.
        lo: The current lower bound, prefilled in the lower spin box.
        hi: The current upper bound, prefilled in the upper spin box.
        hint: An optional gray hint line shown above the spin boxes.
        parent: The parent widget; defaults to None.
    """

    def __init__(
        self,
        title: str,
        lo: float,
        hi: float,
        hint: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(340)

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        if hint is not None:
            hint_label: QLabel = QLabel(hint)
            hint_label.setWordWrap(True)
            hint_label.setStyleSheet("color: gray;")
            layout.addWidget(hint_label)

        rows: list[tuple[str, QDoubleSpinBox]] = []
        for caption, bound in (("Lower bound", lo), ("Upper bound", hi)):
            row: QHBoxLayout = QHBoxLayout()
            row.setSpacing(6)
            caption_label: QLabel = QLabel(caption)
            spin: QDoubleSpinBox = QDoubleSpinBox()
            spin.setRange(_BOUND_RANGE_MIN, _BOUND_RANGE_MAX)
            spin.setDecimals(_BOUND_DECIMALS)
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

        self._min_spin.valueChanged.connect(self._validate)
        self._max_spin.valueChanged.connect(self._validate)
        self._validate(0.0)

    def run(self) -> tuple[float, float] | None:
        """
        Show the dialog modally and collect the user's choice.

        Returns:
            The (lower, upper) bounds the user accepted, or None
            when the dialog was cancelled.
        """
        if self.exec() == QDialog.DialogCode.Accepted:
            return (self._min_spin.value(), self._max_spin.value())
        return None

    def _validate(self, _value: float) -> None:
        """Enable OK only while the two bounds form a strict range."""
        self._ok_button.setEnabled(self._min_spin.value() < self._max_spin.value())


class ThemedGroup(QWidget):
    """
    A collapsible, optionally colored parameter-group container.

    The header is a checkable, auto-raise tool button carrying the
    group's title and a collapse arrow; the content widget holding
    the group's cells is shown or hidden with the button.  A group
    built with a `Theme` gets the theme's light background, an
    accent top border, and an accent-colored bold title; the
    neutral group (the plot selection) is built with `theme=None`
    and keeps the default palette.

    The content is a single vertical column by default.  A group
    built with `grid_columns` lays its cells out in a grid with that
    many columns, read left to right, top to bottom: each cell takes
    one column, and a header added with `add_header` starts a new
    row and spans all of the columns.

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
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(2)

        self._toggle: QToolButton = QToolButton(self)
        self._toggle.setCheckable(True)
        self._toggle.setAutoRaise(True)
        self._toggle.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self._content: QWidget = QWidget(self)
        if grid_columns is not None:
            grid: QGridLayout = QGridLayout(self._content)
            grid.setContentsMargins(8, 2, 8, 8)
            grid.setSpacing(4)
            for column_index in range(grid_columns):
                grid.setColumnStretch(column_index, 1)
            grid.setRowStretch(0, 1)
            self._content_layout: QGridLayout | QVBoxLayout = grid
        else:
            column: QVBoxLayout = QVBoxLayout(self._content)
            column.setContentsMargins(8, 2, 8, 8)
            column.addStretch()
            self._content_layout = column

        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

        if theme is not None:
            self.setObjectName(f"param_group_{id(self)}")
            self.setStyleSheet(
                f"#param_group_{id(self)} {{"
                f" background-color: {theme.light};"
                f" border-top: 3px solid {theme.accent};"
                " }"
                "QToolButton {"
                f" color: {theme.accent};"
                " font-weight: bold; border: none;"
                " padding: 4px 0; text-align: left;"
                " }"
            )

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
        self._grid_pos += span
        if self._grid_stretch_row <= row:
            grid.setRowStretch(self._grid_stretch_row, 0)
            self._grid_stretch_row = row + 1
            grid.setRowStretch(self._grid_stretch_row, 1)

    def replace_widget(self, old: QWidget, new: QWidget) -> None:
        """
        Replace `old` with `new` at the same position in the content.

        The old widget is removed from the layout and scheduled for
        deletion.

        Args:
            old: The widget to replace (a member of the content).
            new: The widget that takes its place.

        Raises:
            ValueError: Raised when `old` is not in the content
                layout.
        """
        if isinstance(self._content_layout, QGridLayout):
            assert self._grid_columns is not None
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
            _ARROW_EXPANDED if self._toggle.isChecked() else _ARROW_COLLAPSED
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
    figure-selection checkboxes.

    The groups, in order, are: *Hydrology* (blue-themed; one cell
    per zone parameter in `ZONE_ORDER`), *Catchment proportion*
    (the single area-fraction parameter), *Reactive transport*
    (red-themed; per-zone storage parameters and per-mineral kinetic
    parameters in every non-snow zone), *Initial state* (red-themed;
    the calibrated exchange-site state per zone), and *Plots*
    (the species / zone / mineral checkboxes for the zone and
    reaction-rate figures).

    The four parameter groups lay their cells out in a four-column
    grid: a cell's symbol and spin box take one column each, so a
    row holds two cells.  Free parameters (distinct bounds) are
    editable spin-box cells; degenerate parameters (identical
    bounds) are pinned and shown as read-only cells that unlock on
    double-click.  A free cell's range can be re-edited by
    double-clicking the cell.

    Signals:
        value_changed: Emitted with the parameter key whenever a
            value changes (a spin box was edited, a range was
            edited, or a saved state was restored).
        bounds_changed: Emitted with the parameter key and its new
            (lower, upper) bounds when a range is edited or a pinned
            parameter is unlocked.
        selection_changed: Emitted with the current `PlotSelection`
            whenever a plot checkbox is toggled.

    Args:
        input_data: The loaded calibration inputs (parameter bounds
            and the reaction network).
        selection: The plot selection to start from.
        parent: The parent widget; defaults to None.
    """

    value_changed = Signal(str)
    bounds_changed = Signal(str, float, float)
    selection_changed = Signal(PlotSelection)

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
        self._rows: dict[str, _FreeCell | _PinnedCell] = {}
        self._groups: dict[str, ThemedGroup] = {}

        container: QWidget = QWidget()
        self._container_layout: QVBoxLayout = QVBoxLayout(container)
        self._container_layout.setContentsMargins(8, 8, 8, 8)
        self._container_layout.setSpacing(0)
        self.setWidget(container)

        self._build_param_groups(first_build=True)
        self._plots_group: ThemedGroup = self._build_plots_group(selection)
        self._container_layout.addWidget(self._plots_group)
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
        The current figure selection, as held by the checkbox state.

        Returns:
            The `PlotSelection` of the checked species, zones, and
            minerals, in the network's option order.
        """
        return PlotSelection(
            species=[
                name
                for name in self._input_data.network.species_names
                if self._species_boxes[name].isChecked()
            ],
            zones=[
                zone
                for zone in ZONE_ORDER
                if self._zone_boxes[zone].isChecked()
            ],
            minerals=[
                name
                for name in self._input_data.network.mineral_names
                if self._mineral_boxes[name].isChecked()
            ],
        )

    def set_selection(self, selection: PlotSelection) -> None:
        """
        Set the checkbox state to `selection`.

        The checkboxes' signals are blocked while this runs, so the
        panel does not re-emit `selection_changed`.  Names not in the
        options (a selection saved against a different network) are
        ignored.

        Args:
            selection: The plot selection to apply.
        """
        for name, box in self._species_boxes.items():
            box.blockSignals(True)
            box.setChecked(name in selection.species)
            box.blockSignals(False)
        for zone, box in self._zone_boxes.items():
            box.blockSignals(True)
            box.setChecked(zone in selection.zones)
            box.blockSignals(False)
        for name, box in self._mineral_boxes.items():
            box.blockSignals(True)
            box.setChecked(name in selection.minerals)
            box.blockSignals(False)

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
        Build (or rebuild, after `reset_bounds`) the four parameter
        groups from the current bounds.

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

        self._groups = {
            "hydrology": self._populate_hydrology(),
            "proportion": self._populate_proportion(),
            "reactive": self._populate_reactive(),
            "initial": self._populate_initial(),
        }
        for offset, group in enumerate(self._groups.values()):
            self._container_layout.insertWidget(insert_at + offset, group)

    def _populate_hydrology(self) -> ThemedGroup:
        """The blue-themed hydrology group: one cell per zone parameter."""
        group: ThemedGroup = ThemedGroup(
            "Hydrology", HYDRO_THEME, grid_columns=_GRID_COLUMNS
        )
        for zone in ZONE_ORDER:
            group.add_header(_ZoneHeader(zone))
            for param in HYDRO_PARAMS[zone]:
                self._add_param_cell(group, f"{zone}.{param}")
        return group

    def _populate_proportion(self) -> ThemedGroup:
        """The blue-themed catchment-proportion group: one cell."""
        group: ThemedGroup = ThemedGroup(
            "Catchment proportion", HYDRO_THEME, grid_columns=_GRID_COLUMNS
        )
        self._add_param_cell(group, "proportion.1")
        caption: QLabel = QLabel(
            "Fraction of the catchment area in the distal (hillslope) "
            "column; the riparian column receives the remaining fraction."
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("color: gray;")
        group.add_header(caption)
        return group

    def _populate_reactive(self) -> ThemedGroup:
        """
        The red-themed reactive-transport group: per-zone storage
        parameters, and per-mineral kinetic parameters in every
        non-snow zone.
        """
        group: ThemedGroup = ThemedGroup(
            "Reactive transport", RT_THEME, grid_columns=_GRID_COLUMNS
        )
        mineral_names: list[str] = self._input_data.network.mineral_names
        for zone in ZONE_ORDER:
            group.add_header(_ZoneHeader(zone))
            for param in RT_DIM_PARAMS:
                self._add_param_cell(group, f"{zone}.{param}")
            if zone not in SNOW_ZONES:
                for mineral_id, mineral_name in enumerate(mineral_names):
                    group.add_header(_BoldHeader(mineral_name, top_margin=6))
                    for field in MINERAL_PARAMS:
                        self._add_param_cell(
                            group, f"{zone}.{field}[{mineral_id}]"
                        )
        return group

    def _populate_initial(self) -> ThemedGroup:
        """
        The red-themed initial-state group: the calibrated
        exchange-site state per speciation zone.
        """
        group: ThemedGroup = ThemedGroup(
            "Initial state", RT_THEME, grid_columns=_GRID_COLUMNS
        )
        for zone in SPECIATION_ZONES:
            self._add_param_cell(group, f"initial_state.{zone}_x")
        return group

    def _add_param_cell(self, group: ThemedGroup, key: str) -> None:
        """
        Add the cell for one parameter to `group`, as a free spin-box
        cell or a pinned cell according to its current bounds.
        """
        symbol: str = param_symbol(key)
        label: str = param_label(key, self._input_data.network.mineral_names)
        lo, hi = self._bounds[key]
        if lo == hi:
            pinned_cell: _PinnedCell = _PinnedCell(
                key, symbol, label, self._values[key]
            )
            pinned_cell.unlock_requested.connect(self._on_unlock_requested)
            self._rows[key] = pinned_cell
        else:
            free_cell: _FreeCell = _FreeCell(
                key, symbol, label, (lo, hi), self._values[key], group.theme
            )
            free_cell.value_changed.connect(self._on_value_changed)
            free_cell.edit_requested.connect(self._on_bounds_edit)
            self._rows[key] = free_cell
        group.add_widget(self._rows[key])

    def _build_plots_group(self, selection: PlotSelection) -> ThemedGroup:
        """
        The neutral "Plots" group: the species / zone / mineral
        checkboxes of the zone and reaction-rate figures.

        Args:
            selection: The checkbox state to start from.
        """
        group: ThemedGroup = ThemedGroup("Plots", None)

        group.add_widget(_BoldHeader("Species (zone figure)"))
        self._species_boxes: dict[str, QCheckBox] = {}
        for name in self._input_data.network.species_names:
            box: QCheckBox = QCheckBox(name)
            box.toggled.connect(self._on_selection_changed)
            self._species_boxes[name] = box
            group.add_widget(box)

        group.add_widget(_BoldHeader("Zones (zone figure)"))
        self._zone_boxes: dict[str, QCheckBox] = {}
        for zone in ZONE_ORDER:
            box = QCheckBox(ZONE_LABELS[zone])
            box.toggled.connect(self._on_selection_changed)
            self._zone_boxes[zone] = box
            group.add_widget(box)

        group.add_widget(_BoldHeader("Minerals (rate figure)"))
        self._mineral_boxes: dict[str, QCheckBox] = {}
        for name in self._input_data.network.mineral_names:
            box = QCheckBox(name)
            box.toggled.connect(self._on_selection_changed)
            self._mineral_boxes[name] = box
            group.add_widget(box)

        self.set_selection(selection)
        return group

    def _on_value_changed(self, key: str) -> None:
        """Record a free cell's new value and re-emit it to the window."""
        cell: _FreeCell | _PinnedCell = self._rows[key]
        if isinstance(cell, _FreeCell):
            self._values[key] = cell.value()
        self.value_changed.emit(key)

    def _on_bounds_edit(self, key: str) -> None:
        """
        Open the range dialog for `key` and apply the user's choice
        to both ends of the range.
        """
        label: str = param_label(key, self._input_data.network.mineral_names)
        lo, hi = self._bounds[key]
        decimals: int = bound_decimals(lo, hi)
        hint: str = f"Current range: {lo:.{decimals}f} to {hi:.{decimals}f}."
        result: tuple[float, float] | None = _BoundsDialog(
            f"Edit range — {label}", lo, hi, hint, self
        ).run()
        if result is None:
            return
        new_lo, new_hi = result
        self._bounds[key] = (new_lo, new_hi)

        cell: _FreeCell | _PinnedCell = self._rows[key]
        if not isinstance(cell, _FreeCell):
            return
        cell.set_bounds(new_lo, new_hi)
        self._values[key] = cell.value()

        self.bounds_changed.emit(key, new_lo, new_hi)
        self.value_changed.emit(key)

    def _on_unlock_requested(self, key: str) -> None:
        """
        Open the range dialog for a pinned parameter and, if the user
        accepts a range, convert the cell into a free cell in place.
        """
        label: str = param_label(key, self._input_data.network.mineral_names)
        value: float = self._values[key]
        # A sensible first range around the pinned value: a symmetric
        # span of half the value, or ±1 for a value pinned at zero.
        span: float = abs(value) / 2 if value != 0 else 1.0
        lo, hi = value - span, value + span
        hint: str = (
            f"{label} is pinned to {value:.4f}.  Choose a range to make "
            "it adjustable; the current value is kept."
        )
        result: tuple[float, float] | None = _BoundsDialog(
            f"Edit range — {label}", lo, hi, hint, self
        ).run()
        if result is None:
            return
        new_lo, new_hi = result
        self._bounds[key] = (new_lo, new_hi)
        self._values[key] = min(max(self._values[key], new_lo), new_hi)

        group: ThemedGroup = self._group_of(key)
        free_cell: _FreeCell = _FreeCell(
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
        """Recompute the selection in option order and emit it."""
        self.selection_changed.emit(self.selection())

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
        cell: _FreeCell | _PinnedCell = self._rows[key]
        for group in self._groups.values():
            if cell.parentWidget() is group.content():
                return group
        raise KeyError(f"No parameter group found for {key!r}.")
    # =================== #
# =============== #
