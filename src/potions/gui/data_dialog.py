"""
The column-mapping dialog of the Load data flow.

When the GUI loads a record from a CSV or XLSX file, the file's
columns are mapped onto the record's roles before the loader runs:
the four required data roles (date, precipitation, temperature,
PET), the optional streamflow role, and one optional role per
measured species of the run's reaction network.  The dialog shows
one combo box per row, prefilled from the loader's auto-detection
(`resolve_columns`), and the accepted mapping is passed on to
`load_input_data`, which applies it in `read_spreadsheet` (the date
column) and `resolve_columns` (the data roles).

Values are interpreted in the model's units, without conversion
(the dialog's header states this): precipitation, streamflow, and
PET in mm/day; temperature in °C; concentrations in mol/L.
"""

from typing import Final

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import potions as pt

from potions.gui.data import measured_roles

# The leading items of a row's combo.  A required row's sentinel
# means "not chosen yet" (OK stays disabled until a column is
# picked); an optional row's sentinel means "deliberately absent"
# (the role is excluded from the loaded record).
SELECT_SENTINEL: Final[str] = "(select…)"
NONE_SENTINEL: Final[str] = "(none)"

# The dialog's header: the units policy of the loaded record.
UNITS_POLICY: Final[str] = (
    "Values are used as they appear in the spreadsheet, in the "
    "model's units: precipitation, streamflow, and potential "
    "evapotranspiration in mm/day; temperature in °C; "
    "concentrations in mol/L. No unit conversion is performed."
)


def role_rows(network: pt.ReactionNetwork) -> list[tuple[str, str, bool]]:
    """
    The dialog's rows, in display order.

    The four required data roles first (date, precipitation,
    temperature, PET), then the optional streamflow role and one
    optional row per measured species of the network, in the
    network's plot order.

    Args:
        network: The run's reaction network: it selects the
            measured-species rows.

    Returns:
        ``(role key, display label, required)`` triples.
    """
    rows: list[tuple[str, str, bool]] = [
        ("date", "Date", True),
        ("precip", "Precipitation", True),
        ("temp", "Temperature", True),
        ("pet", "Potential evapotranspiration", True),
        ("q_meas", "Streamflow", False),
    ]
    rows.extend((name, name, False) for name in measured_roles(network))
    return rows


class ColumnMappingDialog(QDialog):
    """
    The role-per-column mapping dialog of the Load data flow.

    One row per record role: the four required data roles, each
    with a leading "(select…)" item, then the optional roles
    (streamflow and the network's measured species), each with a
    leading "(none)" item.  Each row's combo lists the
    spreadsheet's column names, and the values of `prefill` (the
    auto-detected mapping) are applied to the matching rows.  The
    header states the units policy.  OK stays disabled while a
    required row is on "(select…)" or two or more rows point at
    the same column; the status line names the failing constraint
    in each case.

    Args:
        columns: The record's column names, in file order.
        network: The run's reaction network: it selects the
            measured-species rows.
        prefill: The role → column mapping to prefill the rows
            with: a column name picks that item, ``None`` puts the
            row on "(none)", a name that is not in `columns`
            leaves the row on its sentinel, and an absent role
            keeps its sentinel.
        parent: The parent widget; defaults to None.
    """

    def __init__(
        self,
        columns: list[str],
        network: pt.ReactionNetwork,
        prefill: dict[str, str | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Map record columns")
        self._rows: list[tuple[str, str, bool]] = role_rows(network)
        self._combos: dict[str, QComboBox] = {}

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header: QLabel = QLabel(UNITS_POLICY)
        header.setWordWrap(True)
        layout.addWidget(header)

        for role, label, required in self._rows:
            sentinel: str = SELECT_SENTINEL if required else NONE_SENTINEL
            row: QHBoxLayout = QHBoxLayout()
            row.setSpacing(6)
            caption: QLabel = QLabel(label)
            combo: QComboBox = QComboBox()
            combo.addItem(sentinel)
            for name in columns:
                combo.addItem(name)
            if role in prefill:
                value: str | None = prefill[role]
                if value is None:
                    combo.setCurrentIndex(0)
                elif value in columns:
                    combo.setCurrentText(value)
            row.addWidget(caption)
            row.addWidget(combo)
            layout.addLayout(row)
            self._combos[role] = combo

        self._status_label: QLabel = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

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

        for combo in self._combos.values():
            combo.currentIndexChanged.connect(self._validate)
        self._validate(0)

    def mapping(self) -> dict[str, str | None]:
        """
        The user's role → column mapping.

        Every row yields its entry: a required row always yields
        the chosen column name, and an optional row yields None
        when it is on "(none)" (the role is deliberately excluded
        from the loaded record).

        Returns:
            The mapping with one key per row: ``date``, ``precip``,
            ``temp``, ``pet``, ``q_meas``, and the network's
            measured-species names, in the dialog's row order.
        """
        out: dict[str, str | None] = {}
        for role, _label, _required in self._rows:
            text: str = self._combos[role].currentText()
            out[role] = None if text == NONE_SENTINEL else text
        return out

    def _validate(self, _index: int) -> None:
        """
        Live-check the mapping and update the OK button and the
        status line.

        OK is disabled, and the status line explains why, while a
        required row is still on "(select…)" or two or more rows
        point at the same column; otherwise OK is enabled and the
        status line is empty.
        """
        choices: dict[str, str | None] = {}
        for role, combo in self._combos.items():
            text: str = combo.currentText()
            choices[role] = None if text == NONE_SENTINEL else text

        missing: list[str] = [
            label
            for role, label, required in self._rows
            if required and choices[role] == SELECT_SENTINEL
        ]
        if missing:
            self._status_label.setText("Select a column for: " + ", ".join(missing))
            self._ok_button.setEnabled(False)
            return

        labels_by_column: dict[str, list[str]] = {}
        for role, label, _required in self._rows:
            name: str | None = choices[role]
            if name is not None:
                labels_by_column.setdefault(name, []).append(label)
        for column, labels in labels_by_column.items():
            if len(labels) > 1:
                self._status_label.setText(
                    f"Two roles point at the same column {column!r}: "
                    + ", ".join(labels)
                )
                self._ok_button.setEnabled(False)
                return

        self._status_label.setText("")
        self._ok_button.setEnabled(True)
