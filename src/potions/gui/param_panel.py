"""
The parameter panel of the East River lateral-model calibration GUI.

Holds the model's adjustable parameters in the panel below the
figure: the hydrology parameters of every zone (blue-themed), the
single catchment-proportion parameter, the reactive-transport
parameters of every zone (red-themed, with a header per mineral in
the non-snow zones), and the calibrated initial state.

Every parameter whose bounds are distinct is a slider whose range
can be re-edited by double-clicking one of its endpoint labels; the
slider's resolution follows the current range (a thousandth of the
span, see `NUM_SLIDER_STEPS`).  A parameter whose bounds are
identical is pinned and shown as a read-only row; double-clicking
the row opens a dialog that gives it a range, converting it into a
slider.

The last group, "Plots", carries the checkboxes that select the
species, zones, and minerals drawn in the zone and reaction-rate
figures, and emits `selection_changed` whenever one of them is
toggled.
"""

from typing import Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from potions.gui.common import (
    HYDRO_PARAMS,
    HYDRO_THEME,
    MINERAL_PARAMS,
    NUM_SLIDER_STEPS,
    RT_DIM_PARAMS,
    RT_THEME,
    SNOW_ZONES,
    SPECIATION_ZONES,
    Theme,
    ZONE_COLORS,
    ZONE_LABELS,
    ZONE_ORDER,
    format_slider_value,
    param_label,
    slider_decimals,
    step_to_value,
    value_to_step,
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

# The fixed widths (pixels) of the row's trailing labels, so the
# values and the upper endpoints line up across rows.
_VALUE_LABEL_WIDTH: Final[int] = 72
_ENDPOINT_LABEL_WIDTH: Final[int] = 58
# =================== #


# ==== Widgets ==== #
class _EndpointLabel(QLabel):
    """
    A double-clickable min or max endpoint of a slider row's range.

    The label shows one end of the parameter's current range, as a
    hint for the slider's step width.  When double-clicked it emits
    `edit_requested` carrying the position of the end it represents
    (`0` for the minimum, `1` for the maximum), so the panel opens
    the range dialog for just that end.

    Args:
        which: The position of the end shown: `0` for the minimum
            (left-aligned) or `1` for the maximum (right-aligned).
        parent: The parent widget; defaults to None.
    """

    edit_requested = Signal(int)

    def __init__(
        self, which: int, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._which: int = which
        self.setFixedWidth(_ENDPOINT_LABEL_WIDTH)
        alignment: Qt.AlignmentFlag = (
            Qt.AlignmentFlag.AlignLeft
            if which == 0
            else Qt.AlignmentFlag.AlignRight
        )
        self.setAlignment(alignment)
        self.setToolTip("Double-click to edit this end of the range")

    def set_text(self, text: str) -> None:
        """
        Set the text of the endpoint label.

        Args:
            text: The formatted bound to show.
        """
        self.setText(text)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Emit `edit_requested` for this end, then forward the event."""
        self.edit_requested.emit(self._which)
        super().mouseDoubleClickEvent(event)


class _SliderRow(QWidget):
    """
    A single free-parameter row: name, min and max endpoint labels,
    the slider, and the formatted current value.

    The slider spans `NUM_SLIDER_STEPS` integer steps mapped
    linearly onto the (possibly user-edited) range, so re-ranging a
    parameter keeps the slider's resolution at a thousandth of the
    span.  The row emits `value_changed` when the user moves the
    slider and `edit_requested` when an endpoint label is
    double-clicked.

    Signals:
        value_changed: Emitted with the parameter key whenever the
            value changes.
        edit_requested: Emitted with the parameter key and the
            position of the clicked end (`0` for the minimum, `1`
            for the maximum).

    Args:
        key: The parameter's bounds key.
        label: The label shown at the left of the row.
        bounds: The parameter's current (min, max) range.
        value: The parameter's current value, clamped into the range.
        theme: The theme of the group the row belongs to, or None
            for an unthemed group; the slider is tinted with it.
        parent: The parent widget; defaults to None.
    """

    value_changed = Signal(str)
    edit_requested = Signal(str, int)

    def __init__(
        self,
        key: str,
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
        layout.setContentsMargins(8, 1, 8, 1)
        layout.setSpacing(6)

        name_label: QLabel = QLabel(label)
        name_label.setToolTip(label)

        self._min_label: _EndpointLabel = _EndpointLabel(0)
        self._max_label: _EndpointLabel = _EndpointLabel(1)

        self._value_label: QLabel = QLabel()
        self._value_label.setFixedWidth(_VALUE_LABEL_WIDTH)
        self._value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        self._slider: QSlider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, NUM_SLIDER_STEPS)

        layout.addWidget(name_label)
        layout.addWidget(self._min_label)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._value_label)
        layout.addWidget(self._max_label)

        if theme is not None:
            self._apply_theme(theme)

        self._min_label.edit_requested.connect(self._reemit_edit)
        self._max_label.edit_requested.connect(self._reemit_edit)
        self._slider.valueChanged.connect(self._on_slider_moved)

        self._refresh_bounds_labels()
        self.set_value(value)

    def value(self) -> float:
        """The parameter's current value, in the current range."""
        return self._value

    def set_value(self, value: float) -> None:
        """
        Clamp `value` into the current range and move the slider.

        The slider's signals are blocked while the handle moves, so
        this does not re-emit `value_changed`; the caller re-emits
        when the change came from outside the slider (an edited
        range, a restored state).

        Args:
            value: The value to set, clamped into the range.
        """
        lo, hi = self._bounds
        self._value = min(max(value, lo), hi)
        step: int = value_to_step(lo, hi, self._value)
        self._slider.blockSignals(True)
        self._slider.setValue(step)
        self._slider.blockSignals(False)
        self._refresh_value_label()

    def set_bounds(self, lo: float, hi: float) -> None:
        """
        Replace the range, keeping the current (clamped) value.

        Like `set_value`, the slider's signals are blocked, so the
        caller re-emits the resulting changes.

        Args:
            lo: The new lower bound.
            hi: The new upper bound.
        """
        self._bounds = (lo, hi)
        self._refresh_bounds_labels()
        self._value = min(max(self._value, lo), hi)
        step: int = value_to_step(lo, hi, self._value)
        self._slider.blockSignals(True)
        self._slider.setValue(step)
        self._slider.blockSignals(False)
        self._refresh_value_label()

    def _on_slider_moved(self, step: int) -> None:
        """Map a slider step onto the range and emit `value_changed`."""
        lo, hi = self._bounds
        self._value = step_to_value(lo, hi, step)
        self._refresh_value_label()
        self.value_changed.emit(self._key)

    def _reemit_edit(self, which: int) -> None:
        """Forward an endpoint's edit request with this row's key."""
        self.edit_requested.emit(self._key, which)

    def _refresh_bounds_labels(self) -> None:
        """Reformat both endpoint labels for the current range."""
        lo, hi = self._bounds
        decimals: int = slider_decimals(lo, hi)
        self._min_label.set_text(f"{lo:.{decimals}f}")
        self._max_label.set_text(f"{hi:.{decimals}f}")

    def _refresh_value_label(self) -> None:
        """Reformat the value label for the current value and range."""
        lo, hi = self._bounds
        self._value_label.setText(format_slider_value(lo, hi, self._value))

    def _apply_theme(self, theme: Theme) -> None:
        """Tint the slider's groove and handle with the group's theme."""
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal {"
            f" height: 6px; background: {theme.light};"
            " border-radius: 3px; }"
            "QSlider::sub-page:horizontal {"
            f" background: {theme.accent};"
            " border-radius: 3px; }"
            "QSlider::add-page:horizontal { background: transparent; }"
            "QSlider::handle:horizontal {"
            f" background: {theme.accent};"
            " width: 10px; margin: -3px 0; border-radius: 5px; }"
        )


class _FixedRow(QWidget):
    """
    A read-only row for a pinned parameter, whose bounds are
    identical so there is nothing to slide.

    Double-clicking the row emits `unlock_requested`, which the
    panel answers by opening the range dialog and converting the row
    into a slider.

    Args:
        key: The parameter's bounds key.
        label: The label shown at the left of the row.
        value: The pinned value, shown in gray.
        parent: The parent widget; defaults to None.
    """

    unlock_requested = Signal(str)

    def __init__(
        self,
        key: str,
        label: str,
        value: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._key: str = key

        layout: QHBoxLayout = QHBoxLayout(self)
        layout.setContentsMargins(8, 1, 8, 1)
        layout.setSpacing(6)

        name_label: QLabel = QLabel(label)
        name_label.setToolTip(label)

        value_label: QLabel = QLabel(f"{value:.4f}")
        value_label.setFixedWidth(_VALUE_LABEL_WIDTH)
        value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        value_label.setStyleSheet("color: gray;")

        pinned: QLabel = QLabel("(pinned)")
        pinned.setStyleSheet("color: gray;")

        layout.addWidget(name_label)
        layout.addStretch()
        layout.addWidget(value_label)
        layout.addWidget(pinned)

        self.setToolTip(
            "Pinned to a single value. Double-click to give this "
            "parameter a range."
        )

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Emit `unlock_requested` for this row, then forward the event."""
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
    the group's rows is shown or hidden with the button.  A group
    built with a `Theme` gets the theme's light background, an
    accent top border, and an accent-colored bold title; the neutral
    groups (the catchment proportion, the plot selection) are built
    with `theme=None` and keep the default palette.

    Args:
        title: The title shown in the group's header.
        theme: The theme to apply, or None for the neutral groups.
        collapsed: Whether the group starts collapsed.
        parent: The parent widget; defaults to None.
    """

    def __init__(
        self,
        title: str,
        theme: Theme | None,
        collapsed: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title: str = title
        self._theme: Theme | None = theme

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
        self._content_layout: QVBoxLayout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 2, 8, 8)
        self._content_layout.addStretch()

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
        The content widget holding the group's rows.

        Returns:
            The content widget, for membership tests when a row must
            be moved between groups.
        """
        return self._content

    def add_widget(self, widget: QWidget) -> None:
        """
        Insert `widget` into the group's content, above the trailing
        stretch.

        Args:
            widget: The row or header to add.
        """
        self._content_layout.insertWidget(self._content_layout.count() - 1, widget)

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
        index: int = self._content_layout.indexOf(old)
        if index < 0:
            raise ValueError(
                f"Cannot replace a widget that is not in this group's content."
            )
        self._content_layout.removeWidget(old)
        old.deleteLater()
        self._content_layout.insertWidget(index, new)

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

    The groups, in order, are: *Hydrology* (blue-themed; one slider
    per zone parameter in `ZONE_ORDER`), *Catchment proportion*
    (the single area-fraction parameter), *Reactive transport*
    (red-themed; per-zone storage parameters and per-mineral kinetic
    parameters in every non-snow zone), *Initial state* (red-themed;
    the calibrated exchange-site state per zone), and *Plots*
    (the species / zone / mineral checkboxes for the zone and
    reaction-rate figures).

    Free parameters (distinct bounds) are sliders; degenerate
    parameters (identical bounds) are pinned and shown as read-only
    rows that unlock on double-click.  A slider's endpoints can be
    re-edited by double-clicking them.

    Signals:
        value_changed: Emitted with the parameter key whenever a
            value changes (slider moved, range edited, state
            restored).
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
        self._rows: dict[str, _SliderRow | _FixedRow] = {}
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
        by double-clicking a slider's endpoints (or unlocks a pinned
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
        slider again), and the saved values are clamped into the
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
        """The blue-themed hydrology group: one slider per zone parameter."""
        group: ThemedGroup = ThemedGroup("Hydrology", HYDRO_THEME)
        for zone in ZONE_ORDER:
            group.add_widget(_ZoneHeader(zone))
            for param in HYDRO_PARAMS[zone]:
                self._add_param_row(group, f"{zone}.{param}")
        return group

    def _populate_proportion(self) -> ThemedGroup:
        """The blue-themed catchment-proportion group: one slider."""
        group: ThemedGroup = ThemedGroup("Catchment proportion", HYDRO_THEME)
        self._add_param_row(group, "proportion.1")
        caption: QLabel = QLabel(
            "Fraction of the catchment area in the distal (hillslope) "
            "column; the riparian column receives the remaining fraction."
        )
        caption.setWordWrap(True)
        caption.setStyleSheet("color: gray;")
        group.add_widget(caption)
        return group

    def _populate_reactive(self) -> ThemedGroup:
        """
        The red-themed reactive-transport group: per-zone storage
        parameters, and per-mineral kinetic parameters in every
        non-snow zone.
        """
        group: ThemedGroup = ThemedGroup("Reactive transport", RT_THEME)
        mineral_names: list[str] = self._input_data.network.mineral_names
        for zone in ZONE_ORDER:
            group.add_widget(_ZoneHeader(zone))
            for param in RT_DIM_PARAMS:
                self._add_param_row(group, f"{zone}.{param}")
            if zone not in SNOW_ZONES:
                for mineral_id, mineral_name in enumerate(mineral_names):
                    group.add_widget(_BoldHeader(mineral_name, top_margin=6))
                    for field in MINERAL_PARAMS:
                        self._add_param_row(
                            group, f"{zone}.{field}[{mineral_id}]"
                        )
        return group

    def _populate_initial(self) -> ThemedGroup:
        """
        The red-themed initial-state group: the calibrated
        exchange-site state per speciation zone.
        """
        group: ThemedGroup = ThemedGroup("Initial state", RT_THEME)
        for zone in SPECIATION_ZONES:
            self._add_param_row(group, f"initial_state.{zone}_x")
        return group

    def _add_param_row(self, group: ThemedGroup, key: str) -> None:
        """
        Add the row for one parameter to `group`, as a slider or a
        pinned row according to its current bounds.
        """
        label: str = param_label(key, self._input_data.network.mineral_names)
        lo, hi = self._bounds[key]
        if lo == hi:
            fixed_row: _FixedRow = _FixedRow(key, label, self._values[key])
            fixed_row.unlock_requested.connect(self._on_unlock_requested)
            self._rows[key] = fixed_row
        else:
            slider_row: _SliderRow = _SliderRow(
                key, label, (lo, hi), self._values[key], group.theme
            )
            slider_row.value_changed.connect(self._on_value_changed)
            slider_row.edit_requested.connect(self._on_endpoint_edit)
            self._rows[key] = slider_row
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
        """Record a slider's new value and re-emit it to the window."""
        row: _SliderRow | _FixedRow = self._rows[key]
        if isinstance(row, _SliderRow):
            self._values[key] = row.value()
        self.value_changed.emit(key)

    def _on_endpoint_edit(self, key: str, which: int) -> None:
        """
        Open the range dialog for one endpoint of `key` and apply
        the user's choice to that endpoint only.
        """
        label: str = param_label(key, self._input_data.network.mineral_names)
        lo, hi = self._bounds[key]
        decimals: int = slider_decimals(lo, hi)
        end_name: str = "minimum" if which == 0 else "maximum"
        hint: str = (
            f"Current range: {lo:.{decimals}f} to {hi:.{decimals}f}.  "
            f"Edit only the {end_name}; the other end is kept."
        )
        result: tuple[float, float] | None = _BoundsDialog(
            f"Edit range — {label}", lo, hi, hint, self
        ).run()
        if result is None:
            return
        new_lo, new_hi = result
        lo = new_lo if which == 0 else lo
        hi = new_hi if which == 1 else hi
        self._bounds[key] = (lo, hi)

        row: _SliderRow | _FixedRow = self._rows[key]
        if isinstance(row, _SliderRow):
            row.set_bounds(lo, hi)
            self._values[key] = row.value()
        else:
            self._values[key] = min(max(self._values[key], lo), hi)

        self.bounds_changed.emit(key, lo, hi)
        self.value_changed.emit(key)

    def _on_unlock_requested(self, key: str) -> None:
        """
        Open the range dialog for a pinned parameter and, if the user
        accepts a range, convert the row into a slider in place.
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

        old_row: _SliderRow | _FixedRow = self._rows[key]
        group: ThemedGroup = self._group_of(key)
        new_row: _SliderRow = _SliderRow(
            key, label, (new_lo, new_hi), self._values[key], group.theme
        )
        new_row.value_changed.connect(self._on_value_changed)
        new_row.edit_requested.connect(self._on_endpoint_edit)
        self._rows[key] = new_row
        group.replace_widget(old_row, new_row)

        self.bounds_changed.emit(key, new_lo, new_hi)
        self.value_changed.emit(key)

    def _on_selection_changed(self, _checked: bool) -> None:
        """Recompute the selection in option order and emit it."""
        self.selection_changed.emit(self.selection())

    def _group_of(self, key: str) -> ThemedGroup:
        """
        The parameter group containing the row for `key`.

        Args:
            key: The parameter's bounds key.

        Returns:
            The group whose content widget is the row's parent.

        Raises:
            KeyError: Raised when no group contains the row.
        """
        row: _SliderRow | _FixedRow = self._rows[key]
        for group in self._groups.values():
            if row.parentWidget() is group.content():
                return group
        raise KeyError(f"No parameter group found for {key!r}.")
    # =================== #
# =============== #
