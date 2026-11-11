"""
The reaction-network editor dialog of the GUI.

The dialog lets the user author the run's reaction network: which
species it carries (primary aqueous, secondary, mineral,
exchange) and the kinetics of every mineral (one TST or Monod
entry each).  Its working state is a pure-Python `NetworkDraft`
seeded from the active network — the editor never mutates Rust
objects and treats the database as read-only (the source of
"+ Add from database").  After every committed edit the draft is
re-validated and the results refresh the validation bar (errors
block Apply and Save as…).

Apply hands the new network to the caller — the window reloads
exactly as a model switch and writes `network.yaml` when the run
is saved; the dialog itself never writes a file.  *Save as…*
instead writes the draft's canonical YAML to a user-chosen path.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Callable, TypeAlias, TypeVar

import potions as pt
from PySide6.QtCore import QEvent, QObject, Qt  # type: ignore[import-untyped]
from PySide6.QtWidgets import (  # type: ignore[import-untyped]
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QLayoutItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from potions.core import ExchangeReaction
from potions.gui.network_specs import NETWORK_SPECS, get_network_spec
from potions.gui.data import fingerprint_network


@dataclass(frozen=True, slots=True)
class DraftPrimary:
    """A draft primary aqueous species (no stoichiometry)."""

    name: str
    molar_mass: float
    charge: float
    dh_size_param: float


@dataclass(frozen=True, slots=True)
class DraftSecondary:
    """A draft secondary (speciation) species."""

    name: str
    stoichiometry: tuple[tuple[str, float], ...]
    eq_consts: tuple[float, ...]
    molar_mass: float
    charge: float
    dh_size_param: float


@dataclass(frozen=True, slots=True)
class DraftMineral:
    """A draft mineral species."""

    name: str
    molar_mass: float
    stoichiometry: tuple[tuple[str, float], ...]
    eq_consts: tuple[float, ...]
    molar_volume: float


@dataclass(frozen=True, slots=True)
class DraftExchange:
    """A draft exchange (adsorption) reaction."""

    name: str
    stoichiometry: tuple[tuple[str, float], ...]
    log10_k_eq: float
    charge: float


@dataclass(frozen=True, slots=True)
class DraftTst:
    """A draft TST kinetics entry (keyed by its mineral)."""

    mineral_name: str
    label: str
    log10_k: float
    dependence: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class DraftMonod:
    """A draft Monod kinetics entry (keyed by its mineral)."""

    mineral_name: str
    label: str
    log10_k: float
    monod_terms: tuple[tuple[str, float], ...]
    inhib_terms: tuple[tuple[str, float], ...]


# ==== Draft machinery ==== #

_Entry = TypeVar("_Entry")


def _upsert(
    entries: tuple[_Entry, ...],
    entry: _Entry,
    key: Callable[[_Entry], str],
) -> tuple[_Entry, ...]:
    """
    Replace the entry keyed the same as `entry`, or append it.

    Args:
        entries: The section's current entries (ordered, immutable).
        entry: The entry's new value.
        key: Maps an entry to its key — its name, or, for
            kinetics entries, the mineral name.

    Returns:
        The section with the replacement (or append) applied.
    """
    target: str = key(entry)
    updated: list[_Entry] = []
    replaced: bool = False
    for existing in entries:
        if key(existing) == target:
            updated.append(entry)
            replaced = True
        else:
            updated.append(existing)
    if not replaced:
        updated.append(entry)
    return tuple(updated)


def _remove(
    entries: tuple[_Entry, ...],
    name: str,
    key: Callable[[_Entry], str],
) -> tuple[_Entry, ...]:
    """Drop the entry keyed by `name` (a no-op when absent)."""
    return tuple(existing for existing in entries if key(existing) != name)


@dataclass(frozen=True, slots=True)
class NetworkDraft:
    """
    The editable working state of the network editor.

    A frozen mirror of the six `NetworkFile` sections: plain
    strings, floats, and tuples of (name, value) pairs, so the
    editor stays pure Python (headless-testable) and a commit is
    an immutable replacement, not a mutation.  Tuples keep
    insertion order (display order); the canonical YAML sorts
    everything on the way out, so order here is cosmetic.
    """

    primary: tuple[DraftPrimary, ...]
    secondary: tuple[DraftSecondary, ...]
    mineral: tuple[DraftMineral, ...]
    exchange: tuple[DraftExchange, ...]
    tst: tuple[DraftTst, ...]
    monod: tuple[DraftMonod, ...]

    def upsert_primary(self, entry: DraftPrimary) -> "NetworkDraft":
        """Replace or append the primary entry; return the new draft."""
        return replace(self, primary=_upsert(self.primary, entry,
                                             lambda e: e.name))

    def upsert_secondary(self, entry: DraftSecondary) -> "NetworkDraft":
        """Replace or append the secondary entry; return the new draft."""
        return replace(self, secondary=_upsert(self.secondary, entry,
                                               lambda e: e.name))

    def upsert_mineral(self, entry: DraftMineral) -> "NetworkDraft":
        """Replace or append the mineral entry; return the new draft."""
        return replace(self, mineral=_upsert(self.mineral, entry,
                                             lambda e: e.name))

    def upsert_exchange(self, entry: DraftExchange) -> "NetworkDraft":
        """Replace or append the exchange entry; return the new draft."""
        return replace(self, exchange=_upsert(self.exchange, entry,
                                              lambda e: e.name))

    def upsert_tst(self, entry: DraftTst) -> "NetworkDraft":
        """
        Replace or append the TST entry for its mineral; return the
        new draft.  A Monod entry of the same mineral is dropped
        (V5: exactly one kinetics entry per mineral).
        """
        return replace(
            self,
            tst=_upsert(self.tst, entry, lambda e: e.mineral_name),
            monod=_remove(self.monod, entry.mineral_name,
                          lambda e: e.mineral_name),
        )

    def upsert_monod(self, entry: DraftMonod) -> "NetworkDraft":
        """
        Replace or append the Monod entry for its mineral; return
        the new draft.  A TST entry of the same mineral is dropped
        (V5: exactly one kinetics entry per mineral).
        """
        return replace(
            self,
            monod=_upsert(self.monod, entry, lambda e: e.mineral_name),
            tst=_remove(self.tst, entry.mineral_name,
                        lambda e: e.mineral_name),
        )

    def remove_primary(self, name: str) -> "NetworkDraft":
        """Drop the primary entry named `name`; return the new draft."""
        return replace(self, primary=_remove(self.primary, name,
                                             lambda e: e.name))

    def remove_secondary(self, name: str) -> "NetworkDraft":
        """Drop the secondary entry named `name`; return the new draft."""
        return replace(self, secondary=_remove(self.secondary, name,
                                               lambda e: e.name))

    def remove_exchange(self, name: str) -> "NetworkDraft":
        """Drop the exchange entry named `name`; return the new draft."""
        return replace(self, exchange=_remove(self.exchange, name,
                                              lambda e: e.name))

    def remove_mineral(self, name: str) -> "NetworkDraft":
        """
        Drop the mineral named `name` and both its kinetics
        entries; return the new draft.
        """
        return replace(
            self.remove_mineral_kinetics(name),
            mineral=_remove(self.mineral, name, lambda e: e.name),
        )

    def remove_mineral_kinetics(self, name: str) -> "NetworkDraft":
        """Drop both kinetics entries of the mineral `name`."""
        return replace(
            self,
            tst=_remove(self.tst, name, lambda e: e.mineral_name),
            monod=_remove(self.monod, name, lambda e: e.mineral_name),
        )

    @classmethod
    def from_network(cls, network: pt.ReactionNetwork) -> "NetworkDraft":
        """
        Seed a draft from a network, applying N1.

        The canonical-form rule N1 is applied up front so the
        draft never carries references it cannot satisfy:
        stoichiometry terms whose key is not among the carried
        species (the four section names, plus the implicit `X-`
        when the exchange section is non-empty) are dropped —
        exactly what `NetworkFile.from_network` does on the Rust
        side, so a seeded draft and the canonical file of the
        same network agree.  Kinetics terms are copied as-is: N1
        does not touch them (V7, not N1, is their rule).  Both
        are sorted by key for deterministic display.

        Args:
            network: The network to seed from (a preset, a file
                load, or a previously applied draft).

        Returns:
            The draft; sections keep the network's order.
        """
        carried: set[str] = set(network.species_names)
        if network.has_exchange:
            carried.add("X-")

        def _terms(terms: dict[str, float]) -> tuple[tuple[str, float], ...]:
            return tuple(
                (key, value)
                for key, value in sorted(terms.items())
                if key in carried
            )

        kinetics: pt.MineralKineticData = network.mineral_kinetics
        return cls(
            primary=tuple(
                DraftPrimary(
                    name=s.name,
                    molar_mass=s.molar_mass,
                    charge=s.charge,
                    dh_size_param=s.dh_size_param,
                )
                for s in network.primary_aqueous
            ),
            secondary=tuple(
                DraftSecondary(
                    name=s.name,
                    stoichiometry=_terms(s.stoichiometry),
                    eq_consts=tuple(s.eq_consts),
                    molar_mass=s.molar_mass,
                    charge=s.charge,
                    dh_size_param=s.dh_size_param,
                )
                for s in network.secondary
            ),
            mineral=tuple(
                DraftMineral(
                    name=s.name,
                    molar_mass=s.molar_mass,
                    stoichiometry=_terms(s.stoichiometry),
                    eq_consts=tuple(s.eq_consts),
                    molar_volume=s.molar_volume,
                )
                for s in network.mineral
            ),
            exchange=tuple(
                DraftExchange(
                    name=s.name,
                    stoichiometry=_terms(s.stoichiometry),
                    log10_k_eq=s.log10_k_eq,
                    charge=s.charge,
                )
                for s in network.exchange_species
            ),
            tst=tuple(
                DraftTst(
                    mineral_name=r.mineral_name,
                    label=r.label,
                    log10_k=r.rate_constant,
                    dependence=tuple(sorted(r.dependence.items())),
                )
                for r in sorted(kinetics.tst_reactions.values(),
                                key=lambda r: r.mineral_name)
            ),
            monod=tuple(
                DraftMonod(
                    mineral_name=r.mineral_name,
                    label=r.label,
                    log10_k=r.rate_constant,
                    monod_terms=tuple(sorted(r.monod_terms.items())),
                    inhib_terms=tuple(sorted(r.inhib_terms.items())),
                )
                for r in sorted(kinetics.monod_reactions.values(),
                                key=lambda r: r.mineral_name)
            ),
        )

    def to_network(self) -> pt.ReactionNetwork:
        """
        Build the network this draft describes.

        The constructor is lenient on purpose: it accepts dangling
        references (the runtime reads absent keys as 0.0) and name
        collisions; `draft_validation` — not this method — is the
        gate.  Sections keep the draft's order.

        Returns:
            The new `ReactionNetwork`.
        """
        return pt.ReactionNetwork(
            primary_aqueous=[
                pt.PrimaryAqueousSpecies(
                    p.name, p.molar_mass, p.charge, p.dh_size_param
                )
                for p in self.primary
            ],
            secondary=[
                pt.SecondarySpecies(
                    s.name,
                    dict(s.stoichiometry),
                    list(s.eq_consts),
                    s.molar_mass,
                    s.charge,
                    s.dh_size_param,
                )
                for s in self.secondary
            ],
            mineral=[
                pt.MineralSpecies(
                    m.name,
                    m.molar_mass,
                    dict(m.stoichiometry),
                    list(m.eq_consts),
                    m.molar_volume,
                )
                for m in self.mineral
            ],
            mineral_kinetics=pt.MineralKineticData(
                tst_reactions={
                    t.mineral_name: pt.TstReaction(
                        t.mineral_name, t.label, t.log10_k,
                        dict(t.dependence),
                    )
                    for t in self.tst
                },
                monod_reactions={
                    m.mineral_name: pt.MonodReaction(
                        m.mineral_name,
                        m.label,
                        m.log10_k,
                        dict(m.monod_terms),
                        dict(m.inhib_terms),
                    )
                    for m in self.monod
                },
            ),
            exchange_species=[
                ExchangeReaction(
                    e.name,
                    dict(e.stoichiometry),
                    e.log10_k_eq,
                    e.charge,
                )
                for e in self.exchange
            ],
        )


def _dangling_reference_issues(draft: NetworkDraft) -> list[str]:
    """
    V3/V7-style findings on the draft's un-normalized entries.

    The network-level `validate()` (T2) runs after N1, which drops
    exactly these keys, so dangling references are checked here,
    on the raw draft, before anything is normalized.  A key
    dangles when it is not among the carried species — the four
    section names, plus the implicit `X-` when the exchange
    section is non-empty.

    Args:
        draft: The draft to check.

    Returns:
        The findings, each prefixed with its rule code and naming
        the offender; empty when nothing dangles.
    """
    carried: set[str] = {
        entry.name
        for section in (
            draft.primary,
            draft.secondary,
            draft.mineral,
            draft.exchange,
        )
        for entry in section
    }
    if draft.exchange:
        carried.add("X-")

    issues: list[str] = []
    for secondary in draft.secondary:
        for key, _ in secondary.stoichiometry:
            if key not in carried:
                issues.append(
                    f"V3: stoichiometry of secondary species "
                    f"'{secondary.name}' references unknown species '{key}'"
                )
    for mineral in draft.mineral:
        for key, _ in mineral.stoichiometry:
            if key not in carried:
                issues.append(
                    f"V3: stoichiometry of mineral '{mineral.name}' "
                    f"references unknown species '{key}'"
                )
    for exchange in draft.exchange:
        for key, _ in exchange.stoichiometry:
            if key not in carried:
                issues.append(
                    f"V3: stoichiometry of exchange species "
                    f"'{exchange.name}' references unknown species '{key}'"
                )
    for tst in draft.tst:
        for key, _ in tst.dependence:
            if key not in carried:
                issues.append(
                    f"V7: TST '{tst.label}' of {tst.mineral_name} "
                    f"references unknown species '{key}'"
                )
    for monod in draft.monod:
        for kind, terms in (
            ("monod_terms", monod.monod_terms),
            ("inhib_terms", monod.inhib_terms),
        ):
            for key, _ in terms:
                if key not in carried:
                    issues.append(
                        f"V7: {kind} of Monod '{monod.label}' of "
                        f"{monod.mineral_name} references unknown "
                        f"species '{key}'"
                    )
    return issues


def draft_validation(
    draft: NetworkDraft,
) -> tuple[list[str], list[str]]:
    """
    The draft's validation findings, split into errors and warnings.

    Two passes (the module's design note): the raw-draft pass
    catches dangling references (V3/V7) before N1 can normalize
    them away; the network pass runs the full T2 validation on
    the built network (V1–V9, W1/W2).  The findings are merged,
    so one edit surfaces every problem at once.

    Args:
        draft: The draft to validate.

    Returns:
        (errors, warnings): human-readable messages, each
        prefixed with its rule code.  The first is empty when the
        draft may be applied; the second never blocks.
    """
    issues: list[pt.NetworkValidationIssue] = draft.to_network().validate()
    errors: list[str] = _dangling_reference_issues(draft)
    warnings: list[str] = []
    for issue in issues:
        text: str = f"{issue.code}: {issue.message}"
        if issue.code.startswith("V"):
            errors.append(text)
        else:
            warnings.append(text)
    return errors, warnings


# ==== The dialog ==== #

# The four species sections the dialog edits, in display order.
_SECTIONS: tuple[str, ...] = ("primary", "secondary", "mineral",
                              "exchange")

_SECTION_TITLES: dict[str, str] = {
    "primary": "Primary aqueous",
    "secondary": "Secondary (speciation)",
    "mineral": "Mineral",
    "exchange": "Exchange",
}

_PRESET_FINGERPRINT_MAP: dict[str, str] | None = None


def _preset_fingerprint_map() -> dict[str, str]:
    """
    The fingerprint-to-key map of the preset networks.

    Built lazily once per process: each preset builder loads the
    default database, so opening the dialog pays that cost at most
    once.  The fingerprint is order-independent (T7), so the map is
    stable against the database order the presets build in.

    Returns:
        The map of preset fingerprint to preset key.
    """
    global _PRESET_FINGERPRINT_MAP
    if _PRESET_FINGERPRINT_MAP is None:
        _PRESET_FINGERPRINT_MAP = {
            fingerprint_network(spec.builder()): spec.key
            for spec in NETWORK_SPECS.values()
        }
    return _PRESET_FINGERPRINT_MAP


def _make_spin(low: float, high: float, decimals: int) -> QDoubleSpinBox:
    """
    A double spin box over a generous range.

    The range only guards against nonsense input — validation, not
    the widgets, is the gate.

    Args:
        low: The minimum value the box accepts.
        high: The maximum value the box accepts.
        decimals: The number of decimal places shown and edited.

    Returns:
        The configured spin box.
    """
    spin: QDoubleSpinBox = QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setDecimals(decimals)
    return spin


def _draft_carried(draft: NetworkDraft) -> set[str]:
    """
    The species a draft's stoichiometry may reference.

    The four sections' names, plus the implicit ``X-`` when the
    exchange section is non-empty — the draft-level mirror of the
    V3 set (and of `NetworkFile`'s `carried_names` on the Rust
    side).
    """
    carried: set[str] = {
        entry.name
        for section in (
            draft.primary,
            draft.secondary,
            draft.mineral,
            draft.exchange,
        )
        for entry in section
    }
    if draft.exchange:
        carried.add("X-")
    return carried


# The detail form's rows, in the order `_build_detail_editor`
# adds them (form rows 0..8); `_fill_detail` shows the subset
# each entry type has.
_DETAIL_ROW_KEYS: tuple[str, ...] = (
    "name",
    "molar_mass",
    "charge",
    "dh_size",
    "molar_volume",
    "log10_k_eq",
    "eq_consts",
    "stoichiometry",
    "buttons",
)

# The rows each entry type shows; the rest stay hidden.
_ROWS_VISIBLE_BY_SECTION: dict[str, tuple[str, ...]] = {
    "primary": ("name", "molar_mass", "charge", "dh_size"),
    "secondary": (
        "name",
        "molar_mass",
        "charge",
        "dh_size",
        "eq_consts",
        "stoichiometry",
        "buttons",
    ),
    "mineral": (
        "name",
        "molar_mass",
        "molar_volume",
        "eq_consts",
        "stoichiometry",
        "buttons",
    ),
    "exchange": (
        "name",
        "charge",
        "log10_k_eq",
        "stoichiometry",
        "buttons",
    ),
}

# The entry shapes the detail pane can edit: all four draft
# types, and the subset that carries a stoichiometry table
# (the primary types have none).
_SpeciesEntry: TypeAlias = (
    DraftPrimary | DraftSecondary | DraftMineral | DraftExchange
)
_StoichEntry: TypeAlias = DraftSecondary | DraftMineral | DraftExchange


# (log10 k, TST dependence, Monod terms, inhib terms) seeded from
# the database's reaction of the same mineral/kind/label; the
# list not matching the kind is empty.
_SeededKinetic: TypeAlias = tuple[
    float,
    list[tuple[str, float]],
    list[tuple[str, float]],
    list[tuple[str, float]],
]


def _kinetics_term_keys(
    entry: DraftTst | DraftMonod | None,
) -> set[str]:
    """
    The species keys a kinetics entry's term table references.

    TST entries carry a flat dependence; Monod entries carry the
    monod and inhibitation keys together.  A mineral without a
    kinetics entry (yet) contributes nothing.
    """
    if entry is None:
        return set()
    if isinstance(entry, DraftTst):
        return {key for key, _ in entry.dependence}
    return {key for key, _ in entry.monod_terms} | {
        key for key, _ in entry.inhib_terms
    }


class _FocusOutFilter(QObject):  # type: ignore[misc]
    """
    Invokes a commit callback when the watched widget loses
    keyboard focus.

    PySide6 does not expose the focus-in / focus-out signals
    that the C++ widget API documents, so an event filter is
    the direct way to commit an edit on focus-out without
    subclassing the widget.  The filter is parented to the
    widget, so it is destroyed with it.

    Args:
        target:
            The widget whose focus loss triggers the commit.

        commit:
            The zero-argument callback to run on focus-out.
    """

    def __init__(self, target: QWidget, commit: Callable[[], None]) -> None:
        super().__init__(target)
        self._target: QWidget = target
        self._commit: Callable[[], None] = commit

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._target and event.type() is QEvent.Type.FocusOut:
            self._commit()
        return False


class NetworkEditorDialog(QDialog):  # type: ignore[misc]
    """
    The modal dialog that authors the run's reaction network.

    The dialog's working state is a pure-Python `NetworkDraft`
    seeded from the network it was given.  The left pane lists the
    database's entries for each of the four species sections, plus
    draft-only entries marked ``✚``; an entry is carried by the
    draft exactly when its checkbox is on.  The right pane edits
    the selected entry's fields and stoichiometry; below the
    splitter, one row per carried mineral edits that mineral's
    kinetics (exactly one TST or Monod entry each).  The validation
    bar re-runs the draft's validation after every committed edit:
    errors disable Apply and Save as…, warnings only annotate.

    Apply hands the resulting `ReactionNetwork` to the caller
    through the `network` property without writing a file — the
    window writes `network.yaml` into the run directory when the
    run is saved.  *Save as…* writes the draft's canonical YAML to
    a user-chosen path and the dialog stays open.

    Args:
        network: The network the draft starts from (a preset, a
            file load, or a previously applied draft).
        database: The default chemical database — the source of
            the browser entries and of "+ Add from database".
            Treated as read-only.
        parent: The dialog's parent widget; defaults to None.
    """

    def __init__(
        self,
        network: pt.ReactionNetwork,
        database: pt.ChemicalDatabase,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._database: pt.ChemicalDatabase = database
        self._seed_network: pt.ReactionNetwork = network
        self._network: pt.ReactionNetwork | None = None
        self._dirty: bool = False
        # Guards the checkbox handlers while the browsers rebuild.
        self._rebuilding: bool = False
        # (section, entry name) shown in the detail pane; None until
        # the first selection.
        self._selected: tuple[str, str] | None = None
        # The per-mineral kinetics row widgets;
        # `_rebuild_kinetics_rows` rebuilds all four dicts
        # together, so a `deleteLater`-ed row can never deliver a
        # late event into a stale entry (the handlers guard on
        # `_rebuilding` as well).  The label field is a combo when
        # the database offers several labels for the
        # mineral/kind, a line edit otherwise; the commit handler
        # branches on the concrete type.
        self._kinetics_type: dict[str, QComboBox] = {}
        self._kinetics_label: dict[str, QWidget] = {}
        self._kinetics_k: dict[str, QDoubleSpinBox] = {}
        self._kinetics_terms: dict[str, QTableWidget] = {}

        self.setWindowTitle("Edit reaction network")
        self.setMinimumSize(900, 640)

        layout: QVBoxLayout = QVBoxLayout(self)
        layout.addLayout(self._build_top_row())
        splitter: QSplitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_browsers())
        splitter.addWidget(self._build_detail_editor())
        splitter.setSizes([360, 540])
        layout.addWidget(splitter, 1)
        layout.addWidget(self._build_kinetics_panel())
        layout.addWidget(self._build_validation_bar())
        layout.addLayout(self._build_button_row())

        self._draft: NetworkDraft = NetworkDraft.from_network(network)
        self._reload_panes()

    def _build_top_row(self) -> QHBoxLayout:
        """
        The top row: the preset combo and the two entry-creation
        buttons.

        The combo opens on the preset whose fingerprint matches the
        seeded network's (order-independent, T7), unselected when
        the network is custom.
        """
        row: QHBoxLayout = QHBoxLayout()
        row.addWidget(QLabel("Network:"))
        self._preset_combo: QComboBox = QComboBox()
        for key, spec in NETWORK_SPECS.items():
            self._preset_combo.addItem(spec.display_name, key)
        initial: str | None = _preset_fingerprint_map().get(
            fingerprint_network(self._seed_network)
        )
        self._preset_index: int = -1
        for index in range(self._preset_combo.count()):
            if self._preset_combo.itemData(index) == initial:
                self._preset_combo.setCurrentIndex(index)
                self._preset_index = index
                break
        if self._preset_index == -1:
            self._preset_combo.setCurrentIndex(-1)
        self._preset_combo.currentIndexChanged.connect(
            self._on_preset_changed
        )
        row.addWidget(self._preset_combo)
        self._add_button: QPushButton = QPushButton("+ Add from database")
        self._add_button.clicked.connect(self._on_add_from_database)
        row.addWidget(self._add_button)
        self._new_button: QPushButton = QPushButton("New…")
        self._new_button.clicked.connect(self._on_new_entry)
        row.addWidget(self._new_button)
        row.addStretch(1)
        return row

    def _build_browsers(self) -> QWidget:
        """
        The left column: one searchable, checkable list per section.

        Each list shows the database's entries for its section
        (sorted names) plus the draft's own entries that are not in
        the database, marked ``✚`` in insertion order.  Checked
        means "carried by the draft" (the `_rebuild_browsers`
        handler, step 4, keeps them in sync).
        """
        container: QWidget = QWidget()
        column: QVBoxLayout = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        self._browser_search: dict[str, QLineEdit] = {}
        self._browser_list: dict[str, QListWidget] = {}
        for section in _SECTIONS:
            box: QGroupBox = QGroupBox(_SECTION_TITLES[section])
            inner: QVBoxLayout = QVBoxLayout(box)
            search: QLineEdit = QLineEdit()
            search.setPlaceholderText("Search…")
            search.textChanged.connect(
                lambda _text, s=section: self._filter_browser(s)
            )
            inner.addWidget(search)
            list_widget: QListWidget = QListWidget()
            list_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            list_widget.setSelectionMode(
                QListWidget.SelectionMode.SingleSelection
            )
            list_widget.itemSelectionChanged.connect(
                lambda s=section: self._on_browser_selection(s)
            )
            list_widget.itemChanged.connect(
                lambda _item, s=section: self._on_checkbox_toggled(s)
            )
            inner.addWidget(list_widget)
            self._browser_search[section] = search
            self._browser_list[section] = list_widget
            column.addWidget(box)
        return container

    def _build_detail_editor(self) -> QWidget:
        """
        The right pane: the selected entry's fields.

        Every field of every entry type is built; `_populate_detail`
        (step 4) shows only the rows the selected entry's type has
        and fills them.  The stoichiometry table holds one
        (species combo, coefficient spin) widget pair per draft
        row, rebuilt by `_populate_detail` (step 4); the add/remove
        row buttons are wired there as well.
        """
        container: QWidget = QWidget()
        column: QVBoxLayout = QVBoxLayout(container)
        self._detail_title: QLabel = QLabel("")
        self._detail_title.setStyleSheet("font-weight: bold;")
        column.addWidget(self._detail_title)

        scroll: QScrollArea = QScrollArea()
        scroll.setWidgetResizable(True)
        body: QWidget = QWidget()
        form: QFormLayout = QFormLayout(body)
        self._detail_form: QFormLayout = form

        self._name_edit: QLineEdit = QLineEdit()
        self._name_edit.returnPressed.connect(self._commit_name)
        self._name_edit.installEventFilter(
            _FocusOutFilter(self._name_edit, self._commit_name)
        )
        form.addRow("Name", self._name_edit)

        self._molar_mass_spin: QDoubleSpinBox = _make_spin(0.0, 1e6, 6)
        self._charge_spin: QDoubleSpinBox = _make_spin(-10.0, 10.0, 4)
        self._dh_size_spin: QDoubleSpinBox = _make_spin(0.0, 100.0, 6)
        self._molar_volume_spin: QDoubleSpinBox = _make_spin(0.0, 100.0, 6)
        self._log10_k_spin: QDoubleSpinBox = _make_spin(-30.0, 30.0, 6)
        for spin in (
            self._molar_mass_spin,
            self._charge_spin,
            self._dh_size_spin,
            self._molar_volume_spin,
            self._log10_k_spin,
        ):
            spin.valueChanged.connect(self._commit_detail_fields)
        form.addRow("Molar mass", self._molar_mass_spin)
        form.addRow("Charge", self._charge_spin)
        form.addRow("DH size", self._dh_size_spin)
        form.addRow("Molar volume", self._molar_volume_spin)
        form.addRow("log10 K_eq", self._log10_k_spin)

        eq_grid: QGridLayout = QGridLayout()
        self._eq_spins: list[QDoubleSpinBox] = [
            _make_spin(-100.0, 100.0, 6) for _ in range(8)
        ]
        for index, spin in enumerate(self._eq_spins):
            spin.valueChanged.connect(self._commit_detail_fields)
            eq_grid.addWidget(spin, index // 4, index % 4)
        form.addRow("Eq. consts", eq_grid)

        self._stoich_table: QTableWidget = QTableWidget(0, 2)
        self._stoich_table.setHorizontalHeaderLabels(["Species", "Coeff."])
        stoich_buttons: QHBoxLayout = QHBoxLayout()
        self._stoich_add_button: QPushButton = QPushButton("Add row")
        self._stoich_add_button.clicked.connect(self._on_stoich_add)
        stoich_buttons.addWidget(self._stoich_add_button)
        self._stoich_remove_button: QPushButton = QPushButton(
            "Remove row"
        )
        self._stoich_remove_button.clicked.connect(
            self._on_stoich_remove
        )
        stoich_buttons.addWidget(self._stoich_remove_button)
        form.addRow("Stoichiometry", self._stoich_table)
        form.addRow("", stoich_buttons)

        scroll.setWidget(body)
        column.addWidget(scroll, 1)
        return container

    def _build_kinetics_panel(self) -> QWidget:
        """
        The kinetics panel: one row per carried mineral (TST/Monod
        type combo, label field, log10 k, term table).  The panel is
        an empty scroll area here; `_rebuild_kinetics_rows` (step
        4) fills it from the draft and tears the rows down on
        change.
        """
        scroll: QScrollArea = QScrollArea()
        scroll.setWidgetResizable(True)
        body: QWidget = QWidget()
        self._kinetics_layout: QVBoxLayout = QVBoxLayout(body)
        self._kinetics_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(body)
        return scroll

    def _build_validation_bar(self) -> QWidget:
        """
        The three-line status bar under the splitter.

        The green line is the entry counts, the red line the
        blocking findings (disables Apply and Save as…), the amber
        line the non-blocking warnings.  `_refresh_validation`
        keeps exactly the non-empty lines visible.
        """
        container: QWidget = QWidget()
        column: QVBoxLayout = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        self._ok_label: QLabel = QLabel()
        self._ok_label.setWordWrap(True)
        self._ok_label.setStyleSheet("color: #1e7e34;")
        self._error_label: QLabel = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #c0392b;")
        self._warning_label: QLabel = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #b7950b;")
        for label in (
            self._ok_label,
            self._error_label,
            self._warning_label,
        ):
            column.addWidget(label)
        return container

    def _build_button_row(self) -> QHBoxLayout:
        """The bottom row: Cancel / Save as… / Apply, right-aligned."""
        row: QHBoxLayout = QHBoxLayout()
        row.addStretch(1)
        self._cancel_button: QPushButton = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel_button)
        self._save_as_button: QPushButton = QPushButton("Save as…")
        self._save_as_button.clicked.connect(self._on_save_as)
        row.addWidget(self._save_as_button)
        self._apply_button: QPushButton = QPushButton("Apply")
        self._apply_button.clicked.connect(self._on_apply)
        row.addWidget(self._apply_button)
        return row

    @property
    def network(self) -> pt.ReactionNetwork | None:
        """
        The applied network; ``None`` until the user clicks Apply.

        The window reads this after `edit_network` returns an
        accepted dialog.
        """
        return self._network

    def _reload_panes(self) -> None:
        """
        Rebuild every pane from the current draft.

        Used after the initial seed and after a preset re-seed;
        checkbox toggles use the lighter per-pane refreshers
        instead.
        """
        self._rebuild_browsers()
        self._select_first_entry()
        self._rebuild_kinetics_rows()
        self._refresh_validation()

    def _reseed_from_network(self, network: pt.ReactionNetwork) -> None:
        """
        Replace the draft with a fresh seed from `network` and
        reload every pane.

        Clears the dirty flag; the preset combo stays where the
        user put it (the re-seed is its effect).
        """
        self._draft = NetworkDraft.from_network(network)
        self._dirty = False
        self._reload_panes()

    def _on_preset_changed(self, _index: int) -> None:
        """
        Re-seed the draft from the selected preset.

        When the draft carries unapplied edits, confirm first
        (the window's `QMessageBox.question` Yes/No idiom);
        declining restores the combo to its previous selection.
        """
        key: str | None = self._preset_combo.currentData()
        if key is None:
            return
        if self._dirty:
            answer: QMessageBox.StandardButton = QMessageBox.question(
                self,
                "Discard edits?",
                "The draft has unapplied edits. Re-seed it from the "
                f"{self._preset_combo.currentText()} preset?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._preset_combo.blockSignals(True)
                self._preset_combo.setCurrentIndex(self._preset_index)
                self._preset_combo.blockSignals(False)
                return
        self._preset_index = self._preset_combo.currentIndex()
        self._reseed_from_network(get_network_spec(key).builder())

    def _on_cancel(self) -> None:
        """Reject the dialog; the caller keeps its current network."""
        self.reject()

    def _on_save_as(self) -> None:
        """
        Write the draft's canonical YAML to a user-chosen path.

        The dialog stays open; the run's own `network.yaml` is
        written by the window when the run is saved (T10).  The
        button is disabled while the draft carries errors, so this
        only runs on a valid draft.
        """
        path: str = QFileDialog.getSaveFileName(
            self,
            "Save network",
            "network.yaml",
            "YAML files (*.yaml *.yml)",
        )[0]
        if not path:
            return
        try:
            pt.NetworkFile.from_network(self._draft.to_network()).to_file(
                path
            )
        except OSError as error:
            QMessageBox.critical(self, "Save failed", str(error))
            return
        QMessageBox.information(self, "Saved", f"Network written to {path}.")

    def _on_apply(self) -> None:
        """
        Accept the dialog: the draft's network becomes the dialog's
        `network`.

        No file is written here — the window writes `network.yaml`
        into the run directory when the run is saved (T10).  The
        button is disabled while the draft carries errors.
        """
        self._network = self._draft.to_network()
        self._dirty = False
        self.accept()

    def _db_names(self, section: str) -> set[str]:
        """The database's entry names for `section` (read-only)."""
        if section == "primary":
            return set(self._database.primary_species)
        if section == "secondary":
            return set(self._database.secondary_species)
        if section == "mineral":
            return set(self._database.mineral_species)
        return set(self._database.exchange_reactions)

    def _section_in_order(self, section: str) -> list[str]:
        """The draft's names for `section`, in insertion order."""
        if section == "primary":
            return [entry.name for entry in self._draft.primary]
        if section == "secondary":
            return [entry.name for entry in self._draft.secondary]
        if section == "mineral":
            return [entry.name for entry in self._draft.mineral]
        return [entry.name for entry in self._draft.exchange]

    def _draft_names(self, section: str) -> set[str]:
        """The set of the draft's names for `section`."""
        return set(self._section_in_order(section))

    def _rebuild_browsers(self) -> None:
        """
        Rebuild the four browser lists from the draft and the
        database.

        Each list shows the database's entries for its section in
        sorted name order, then the draft's own entries (names not
        in the database) in insertion order, marked ``✚``.  An
        item is checked exactly when its name is in the draft's
        section.
        """
        self._rebuilding = True
        try:
            for section in _SECTIONS:
                list_widget: QListWidget = self._browser_list[section]
                list_widget.blockSignals(True)
                list_widget.clear()
                db_names: set[str] = self._db_names(section)
                draft_names: set[str] = self._draft_names(section)
                entries: list[tuple[str, bool]] = [
                    (name, name in draft_names)
                    for name in sorted(db_names)
                ]
                entries.extend(
                    (name, True)
                    for name in self._section_in_order(section)
                    if name not in db_names
                )
                for name, checked in entries:
                    text: str = name if name in db_names else f"✚ {name}"
                    item: QListWidgetItem = QListWidgetItem(text)
                    item.setFlags(
                        item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if checked
                        else Qt.CheckState.Unchecked
                    )
                    item.setData(Qt.ItemDataRole.UserRole, name)
                    list_widget.addItem(item)
                self._apply_filter(section)
        finally:
            self._rebuilding = False

    def _apply_filter(self, section: str) -> None:
        """Re-apply the section's search-box filter to its list."""
        needle: str = self._browser_search[section].text().strip().lower()
        list_widget: QListWidget = self._browser_list[section]
        for index in range(list_widget.count()):
            item: QListWidgetItem | None = list_widget.item(index)
            assert item is not None
            name: str = str(item.data(Qt.ItemDataRole.UserRole))
            item.setHidden(needle not in name.lower())

    def _filter_browser(self, section: str) -> None:
        """The section's search box changed: re-apply its filter."""
        self._apply_filter(section)

    def _current_browser_name(self, section: str) -> str | None:
        """The name stored in the section's current item (or None)."""
        item: QListWidgetItem | None = self._browser_list[
            section
        ].currentItem()
        if item is None:
            return None
        return str(item.data(Qt.ItemDataRole.UserRole))

    def _on_browser_selection(self, section: str) -> None:
        """The user picked a browser item: show it in the detail pane."""
        if self._rebuilding:
            return
        name: str | None = self._current_browser_name(section)
        if name is None:
            return
        self._selected = (section, name)
        self._populate_detail()

    def _select_first_entry(self) -> None:
        """Select the first draft entry (display order) and show it."""
        self._selected = None
        for section in _SECTIONS:
            names: list[str] = self._section_in_order(section)
            if names:
                self._selected = (section, names[0])
                break
        self._populate_detail()
        self._sync_browser_selection()

    def _sync_browser_selection(self) -> None:
        """Highlight the selected entry's item in its browser list."""
        if self._selected is None:
            return
        section, name = self._selected
        list_widget: QListWidget = self._browser_list[section]
        for index in range(list_widget.count()):
            item: QListWidgetItem | None = list_widget.item(index)
            if (
                item is not None
                and str(item.data(Qt.ItemDataRole.UserRole)) == name
            ):
                list_widget.blockSignals(True)
                list_widget.setCurrentItem(item)
                list_widget.blockSignals(False)
                return

    def _on_checkbox_toggled(self, section: str) -> None:
        """
        A browser checkbox changed: carry or drop that entry.

        Checking a database entry inserts a copy of it (N1'd,
        below); checking a ``✚`` entry is a no-op (it is already in
        the draft — its presence is what keeps it listed).
        Unchecking removes the entry from the draft; for a mineral
        that cascades to its kinetics (V5 keeps the draft valid
        with respect to the removed mineral).
        """
        if self._rebuilding:
            return
        name: str | None = self._current_browser_name(section)
        if name is None:
            return
        item: QListWidgetItem | None = self._browser_list[section].currentItem()
        checked: bool = (
            item is not None
            and item.checkState() == Qt.CheckState.Checked
        )
        if checked and name in self._db_names(section):
            self._upsert_db_entry(section, name)
        elif checked:
            self._after_draft_change()
        else:
            self._remove_draft_entry(section, name)

    def _upsert_db_entry(self, section: str, name: str) -> None:
        """
        Insert the database's `name` entry into the draft's
        `section`, applying N1, then refresh.

        The entry's stoichiometry keys are filtered to the carried
        set *after* the insert (the entry's own name, and ``X-``
        for an exchange entry, are carried once it lands) — the
        same set `NetworkFile.from_network` would keep, so a fresh
        insert can never dangle.  A newly inserted mineral with no
        kinetics yet gets a blank entry of the database's preferred
        kind (TST over Monod); V6 (empty label) keeps it flagged
        until the user names the reaction (step 5's label picker).
        """
        carried: set[str] = _draft_carried(self._draft)
        carried.add(name)
        if section == "exchange":
            carried.add("X-")
        if section == "primary":
            primary: pt.PrimaryAqueousSpecies = (
                self._database.primary_species[name]
            )
            self._draft = self._draft.upsert_primary(
                DraftPrimary(
                    primary.name,
                    primary.molar_mass,
                    primary.charge,
                    primary.dh_size_param,
                )
            )
        elif section == "secondary":
            secondary: pt.SecondarySpecies = (
                self._database.secondary_species[name]
            )
            self._draft = self._draft.upsert_secondary(
                DraftSecondary(
                    secondary.name,
                    tuple(
                        (key, value)
                        for key, value in sorted(
                            secondary.stoichiometry.items()
                        )
                        if key in carried
                    ),
                    tuple(secondary.eq_consts),
                    secondary.molar_mass,
                    secondary.charge,
                    secondary.dh_size_param,
                )
            )
        elif section == "mineral":
            mineral: pt.MineralSpecies = (
                self._database.mineral_species[name]
            )
            self._draft = self._draft.upsert_mineral(
                DraftMineral(
                    mineral.name,
                    mineral.molar_mass,
                    tuple(
                        (key, value)
                        for key, value in sorted(
                            mineral.stoichiometry.items()
                        )
                        if key in carried
                    ),
                    tuple(mineral.eq_consts),
                    mineral.molar_volume,
                )
            )
            has_kinetics: bool = any(
                t.mineral_name == mineral.name for t in self._draft.tst
            ) or any(
                m.mineral_name == mineral.name for m in self._draft.monod
            )
            if not has_kinetics:
                use_monod: bool = (
                    mineral.name not in self._database.tst_reactions
                    and mineral.name in self._database.monod_reactions
                )
                if use_monod:
                    self._draft = self._draft.upsert_monod(
                        DraftMonod(mineral.name, "", 0.0, (), ())
                    )
                else:
                    self._draft = self._draft.upsert_tst(
                        DraftTst(mineral.name, "", 0.0, ())
                    )
        else:
            exchange: ExchangeReaction = (
                self._database.exchange_reactions[name]
            )
            self._draft = self._draft.upsert_exchange(
                DraftExchange(
                    exchange.name,
                    tuple(
                        (key, value)
                        for key, value in sorted(
                            exchange.stoichiometry.items()
                        )
                        if key in carried
                    ),
                    exchange.log10_k_eq,
                    exchange.charge,
                )
            )
        self._after_draft_change()

    def _remove_draft_entry(self, section: str, name: str) -> None:
        """
        Drop the draft's named entry from `section` (a mineral
        removal cascades to its kinetics), then refresh.
        """
        if section == "primary":
            self._draft = self._draft.remove_primary(name)
        elif section == "secondary":
            self._draft = self._draft.remove_secondary(name)
        elif section == "mineral":
            self._draft = self._draft.remove_mineral(name)
        else:
            self._draft = self._draft.remove_exchange(name)
        self._after_draft_change()

    def _after_draft_change(self) -> None:
        """
        The draft changed under the user: mark it dirty, rebuild
        the browsers, keep (or repair) the selection, and refresh
        the kinetics rows and the validation bar.

        The detail pane is re-populated from the (already
        committed) draft; the name field commits on focus-out and
        the spin boxes commit on every change, so nothing the user
        typed is lost.
        """
        self._dirty = True
        self._rebuild_browsers()
        selection: tuple[str, str] | None = self._selected
        if selection is not None:
            section, name = selection
            if name not in self._draft_names(section):
                selection = None
        if selection is None:
            self._select_first_entry()
        else:
            self._populate_detail()
            self._sync_browser_selection()
        self._rebuild_kinetics_rows()
        self._refresh_validation()

    def _selected_entry(
        self, section: str, name: str
    ) -> _SpeciesEntry | None:
        """
        The draft's entry for (section, name).

        None when the draft no longer carries that name (it was
        removed or renamed after the selection was made); the
        caller then clears or repairs the pane.
        """
        section_entries: tuple[_SpeciesEntry, ...]
        if section == "primary":
            section_entries = self._draft.primary
        elif section == "secondary":
            section_entries = self._draft.secondary
        elif section == "mineral":
            section_entries = self._draft.mineral
        else:
            section_entries = self._draft.exchange
        return next(
            (entry for entry in section_entries if entry.name == name),
            None,
        )

    def _populate_detail(self) -> None:
        """
        Render the detail pane for the current selection.

        Runs under the `_rebuilding` guard so filling the widgets
        cannot fire their commit handlers; a missing selection, or
        a selection the draft no longer carries, clears the pane.
        """
        self._rebuilding = True
        try:
            selection: tuple[str, str] | None = self._selected
            entry: _SpeciesEntry | None = None
            if selection is not None:
                entry = self._selected_entry(*selection)
            if selection is None or entry is None:
                self._clear_detail()
            else:
                self._fill_detail(selection[0], entry)
        finally:
            self._rebuilding = False

    def _clear_detail(self) -> None:
        """
        Empty the detail pane (no selection, or the selection left
        the draft).

        Called only under the `_rebuilding` guard, so zeroing the
        widgets commits nothing.
        """
        self._detail_title.setText("")
        self._name_edit.clear()
        for spin in (
            self._molar_mass_spin,
            self._charge_spin,
            self._dh_size_spin,
            self._molar_volume_spin,
            self._log10_k_spin,
            *self._eq_spins,
        ):
            spin.setValue(0.0)
        self._stoich_table.setRowCount(0)

    def _fill_detail(self, section: str, entry: _SpeciesEntry) -> None:
        """
        Fill the detail pane for `entry` of `section`.

        The name is read-only for a database entry (the database's
        identity is its name — renaming only applies to
        draft-only entries) and editable otherwise. Only the rows
        the entry's type has are shown; the field widgets and the
        stoichiometry table are then filled from the entry.
        """
        self._detail_title.setText(
            f"{_SECTION_TITLES[section]} — {entry.name}"
        )
        self._name_edit.setText(entry.name)
        self._name_edit.setReadOnly(entry.name in self._db_names(section))
        visible: tuple[str, ...] = _ROWS_VISIBLE_BY_SECTION[section]
        for index, key in enumerate(_DETAIL_ROW_KEYS):
            self._detail_form.setRowVisible(index, key in visible)
        self._set_detail_fields(entry)
        if isinstance(entry, (DraftSecondary, DraftMineral, DraftExchange)):
            self._rebuild_stoich_rows(entry)
        else:
            self._stoich_table.setRowCount(0)

    def _set_detail_fields(self, entry: _SpeciesEntry) -> None:
        """
        Fill the scalar field widgets from `entry`.

        Each entry type owns a different subset of the fields; the
        rest stay at the values the previous fill (or clear) left
        them, hidden. Called only under the `_rebuilding` guard.
        """
        if isinstance(entry, DraftPrimary):
            self._molar_mass_spin.setValue(entry.molar_mass)
            self._charge_spin.setValue(entry.charge)
            self._dh_size_spin.setValue(entry.dh_size_param)
        elif isinstance(entry, DraftSecondary):
            self._molar_mass_spin.setValue(entry.molar_mass)
            self._charge_spin.setValue(entry.charge)
            self._dh_size_spin.setValue(entry.dh_size_param)
            self._set_eq_consts(entry.eq_consts)
        elif isinstance(entry, DraftMineral):
            self._molar_mass_spin.setValue(entry.molar_mass)
            self._molar_volume_spin.setValue(entry.molar_volume)
            self._set_eq_consts(entry.eq_consts)
        else:
            self._charge_spin.setValue(entry.charge)
            self._log10_k_spin.setValue(entry.log10_k_eq)

    def _set_eq_consts(self, values: Sequence[float]) -> None:
        """
        Fill the eight equation-constant cells from `values`.

        The database's entries all carry exactly eight constants,
        so the fixed grid matches their shape; a shorter tuple (a
        fresh draft entry) shows 0.0 in the spare cells. Called
        only under the `_rebuilding` guard.
        """
        for index, spin in enumerate(self._eq_spins):
            spin.setValue(values[index] if index < len(values) else 0.0)

    def _stoich_options(self, entry: _StoichEntry) -> list[str]:
        """
        The species an entry's stoichiometry rows may reference.

        The draft's carried set (`_draft_carried` — V3's set) plus
        the entry's own current keys. A dangling key (its species
        unchecked) stays selectable, so the row remains editable
        and the V3 finding stays visible: the UI never silently
        normalizes a reference away; fixing it is the user's call
        (re-check the species, or point the row at a carried one).
        """
        own: set[str] = {key for key, _ in entry.stoichiometry}
        return sorted(_draft_carried(self._draft) | own)

    def _rebuild_stoich_rows(self, entry: _StoichEntry) -> None:
        """
        Rebuild the stoichiometry table from `entry`'s rows.

        Each row is a (species combo, coefficient spin) pair; the
        combos offer `_stoich_options`. Every row's widgets are
        built fully before any of them is connected, and the
        caller holds the `_rebuilding` guard, so populating the
        table commits nothing.
        """
        table: QTableWidget = self._stoich_table
        table.setRowCount(0)
        options: list[str] = self._stoich_options(entry)
        for index, (key, value) in enumerate(entry.stoichiometry):
            table.insertRow(index)
            combo: QComboBox = QComboBox()
            combo.addItems(options)
            combo.setCurrentText(key)
            combo.currentIndexChanged.connect(
                lambda _index, row=index: self._commit_stoich_row(row)
            )
            spin: QDoubleSpinBox = _make_spin(-1000.0, 1000.0, 6)
            spin.setValue(value)
            spin.valueChanged.connect(
                lambda _value, row=index: self._commit_stoich_row(row)
            )
            table.setCellWidget(index, 0, combo)
            table.setCellWidget(index, 1, spin)

    def _collect_stoich_rows(self) -> list[tuple[str, float]]:
        """
        The stoichiometry table's rows as (species, coefficient)
        pairs, in table order.

        A row whose widgets are not (yet) attached is skipped —
        possible only mid-rebuild, never for a user-visible table.
        """
        table: QTableWidget = self._stoich_table
        rows: list[tuple[str, float]] = []
        for index in range(table.rowCount()):
            combo: QWidget | None = table.cellWidget(index, 0)
            spin: QWidget | None = table.cellWidget(index, 1)
            if combo is None or spin is None:
                continue
            assert isinstance(combo, QComboBox)
            assert isinstance(spin, QDoubleSpinBox)
            rows.append((combo.currentText(), spin.value()))
        return rows

    def _on_stoich_add(self) -> None:
        """
        Append a stoichiometry row for the first offered species
        (zero coefficient) and commit the table.
        """
        selection: tuple[str, str] | None = self._selected
        if selection is None:
            return
        entry: _SpeciesEntry | None = self._selected_entry(*selection)
        if not isinstance(
            entry, (DraftSecondary, DraftMineral, DraftExchange)
        ):
            return
        options: list[str] = self._stoich_options(entry)
        if not options:
            return
        rows: list[tuple[str, float]] = self._collect_stoich_rows()
        rows.append((options[0], 0.0))
        self._apply_stoich_rows(entry, rows)

    def _on_stoich_remove(self) -> None:
        """
        Drop the last stoichiometry row and commit the table.
        """
        selection: tuple[str, str] | None = self._selected
        if selection is None:
            return
        entry: _SpeciesEntry | None = self._selected_entry(*selection)
        if not isinstance(
            entry, (DraftSecondary, DraftMineral, DraftExchange)
        ):
            return
        rows: list[tuple[str, float]] = self._collect_stoich_rows()
        if not rows:
            return
        rows.pop()
        self._apply_stoich_rows(entry, rows)

    def _apply_stoich_rows(
        self, entry: _StoichEntry, rows: list[tuple[str, float]]
    ) -> None:
        """
        Commit the table's rows into `entry` and refresh.

        The draft is replaced by `entry` carrying the new
        stoichiometry (a tuple, in table order); the table is then
        rebuilt from the committed entry under the `_rebuilding`
        guard, and the validation bar re-runs.
        """
        stoich: tuple[tuple[str, float], ...] = tuple(rows)
        if isinstance(entry, DraftSecondary):
            secondary: DraftSecondary = replace(entry, stoichiometry=stoich)
            updated: _StoichEntry = secondary
            self._draft = self._draft.upsert_secondary(secondary)
        elif isinstance(entry, DraftMineral):
            mineral: DraftMineral = replace(entry, stoichiometry=stoich)
            updated = mineral
            self._draft = self._draft.upsert_mineral(mineral)
        else:
            exchange: DraftExchange = replace(entry, stoichiometry=stoich)
            updated = exchange
            self._draft = self._draft.upsert_exchange(exchange)
        self._dirty = True
        self._rebuilding = True
        try:
            self._rebuild_stoich_rows(updated)
        finally:
            self._rebuilding = False
        self._refresh_validation()

    def _commit_stoich_row(self, _row: int) -> None:
        """
        A stoichiometry row changed (species picked or
        coefficient edited): commit the whole table.

        The row index exists only for the signal's arity; the
        commit works on the table as a whole, so a mid-typing
        state never outlives the event loop.
        """
        if self._rebuilding:
            return
        selection: tuple[str, str] | None = self._selected
        if selection is None:
            return
        entry: _SpeciesEntry | None = self._selected_entry(*selection)
        if not isinstance(
            entry, (DraftSecondary, DraftMineral, DraftExchange)
        ):
            return
        rows: list[tuple[str, float]] = self._collect_stoich_rows()
        self._apply_stoich_rows(entry, rows)

    def _commit_detail_fields(self) -> None:
        """
        A scalar field changed: commit the selected entry's new
        values.

        The commit reads the widgets and replaces the entry in
        the draft; it does not re-populate the pane (that would
        fight a spin mid-typing) and does not rebuild the
        browsers (no name changed).  The eq-consts commit keeps
        the entry's tuple length — the grid is fixed at eight,
        and a shorter entry reads only as many cells as it has.
        """
        if self._rebuilding:
            return
        selection: tuple[str, str] | None = self._selected
        if selection is None:
            return
        entry: _SpeciesEntry | None = self._selected_entry(*selection)
        if entry is None:
            return
        if isinstance(entry, DraftPrimary):
            primary: DraftPrimary = replace(
                entry,
                molar_mass=self._molar_mass_spin.value(),
                charge=self._charge_spin.value(),
                dh_size_param=self._dh_size_spin.value(),
            )
            self._draft = self._draft.upsert_primary(primary)
        elif isinstance(entry, DraftSecondary):
            secondary: DraftSecondary = replace(
                entry,
                molar_mass=self._molar_mass_spin.value(),
                charge=self._charge_spin.value(),
                dh_size_param=self._dh_size_spin.value(),
                eq_consts=tuple(
                    spin.value()
                    for spin in self._eq_spins[: len(entry.eq_consts)]
                ),
            )
            self._draft = self._draft.upsert_secondary(secondary)
        elif isinstance(entry, DraftMineral):
            mineral: DraftMineral = replace(
                entry,
                molar_mass=self._molar_mass_spin.value(),
                molar_volume=self._molar_volume_spin.value(),
                eq_consts=tuple(
                    spin.value()
                    for spin in self._eq_spins[: len(entry.eq_consts)]
                ),
            )
            self._draft = self._draft.upsert_mineral(mineral)
        else:
            exchange: DraftExchange = replace(
                entry,
                charge=self._charge_spin.value(),
                log10_k_eq=self._log10_k_spin.value(),
            )
            self._draft = self._draft.upsert_exchange(exchange)
        self._dirty = True
        self._refresh_validation()

    def _commit_name(self) -> None:
        """
        The name field committed (return or focus-out): rename
        the selected entry.

        Only draft-only entries reach this (a database entry's
        name field is read-only).  A blank name, an unchanged
        name, or a name the same section already carries is
        refused and the field restores its previous text — a
        same-section duplicate would be a key collision in the
        upsert, silently replacing the other entry; a
        cross-section duplicate is allowed here and is V2's job.
        A rename removes and re-inserts the entry under the new
        name (a mineral's kinetics are re-keyed with it); other
        entries' stoichiometry references to the old name are
        deliberately left dangling — a rename is a reference
        break, and V3 must say so, not be normalized (design
        note 2).
        """
        if self._rebuilding:
            return
        selection: tuple[str, str] | None = self._selected
        if selection is None:
            return
        section, old_name = selection
        entry: _SpeciesEntry | None = self._selected_entry(
            section, old_name
        )
        if entry is None:
            return
        new_name: str = self._name_edit.text().strip()
        if not new_name or new_name == old_name:
            self._name_edit.setText(old_name)
            return
        if new_name in self._draft_names(section):
            self._name_edit.setText(old_name)
            return
        if isinstance(entry, DraftPrimary):
            self._draft = (
                self._draft.remove_primary(old_name)
                .upsert_primary(replace(entry, name=new_name))
            )
        elif isinstance(entry, DraftSecondary):
            self._draft = (
                self._draft.remove_secondary(old_name)
                .upsert_secondary(replace(entry, name=new_name))
            )
        elif isinstance(entry, DraftMineral):
            self._rename_mineral(entry, new_name)
        else:
            self._draft = (
                self._draft.remove_exchange(old_name)
                .upsert_exchange(replace(entry, name=new_name))
            )
        self._selected = (section, new_name)
        self._dirty = True
        self._after_draft_change()

    def _rename_mineral(self, entry: DraftMineral, new_name: str) -> None:
        """
        Rename a carried mineral, re-keying its kinetics with it.

        `remove_mineral` cascades to the mineral's kinetics, so a
        bare remove + re-insert would strand them at the old name
        (V5) or drop them; the mineral's own TST/Monod entries
        are therefore re-keyed from the pre-rename draft and
        re-inserted.  (If the draft illegally carried both kinds
        for the mineral, the Monod one wins the re-insert — the
        draft is in error either way.)  References from other
        entries' stoichiometry stay dangling (V3, design note 2).
        """
        old_name: str = entry.name
        original: NetworkDraft = self._draft
        self._draft = (
            original.remove_mineral(old_name)
            .upsert_mineral(replace(entry, name=new_name))
        )
        for tst in original.tst:
            if tst.mineral_name == old_name:
                self._draft = self._draft.upsert_tst(
                    replace(tst, mineral_name=new_name)
                )
        for monod in original.monod:
            if monod.mineral_name == old_name:
                self._draft = self._draft.upsert_monod(
                    replace(monod, mineral_name=new_name)
                )

    def _refresh_validation(self) -> None:
        """
        Re-run the draft's two-stage validation and refresh the
        bar.

        Green line: the entry counts, hidden while any error is
        present.  Red line: the blocking findings — Apply and
        Save as… are enabled only while it is empty.  Amber line:
        the non-blocking warnings (they never disable anything).
        Each label is visible exactly when it carries text.
        """
        errors: list[str]
        warnings: list[str]
        errors, warnings = draft_validation(self._draft)
        n_species: int = (
            len(self._draft.primary)
            + len(self._draft.secondary)
            + len(self._draft.mineral)
            + len(self._draft.exchange)
            + (1 if self._draft.exchange else 0)
        )
        self._ok_label.setText(
            ""
            if errors
            else (
                f"OK — {n_species} species, "
                f"{len(self._draft.mineral)} minerals, "
                f"{len(self._draft.exchange)} exchange"
            )
        )
        self._error_label.setText("\n".join(errors))
        self._warning_label.setText("\n".join(warnings))
        self._ok_label.setVisible(not errors)
        self._error_label.setVisible(bool(errors))
        self._warning_label.setVisible(bool(warnings))
        self._apply_button.setEnabled(not errors)
        self._save_as_button.setEnabled(not errors)

    def _rebuild_kinetics_rows(self) -> None:
        """
        Rebuild the kinetics panel from the draft's minerals.

        One row per carried mineral, in draft order.  Every
        previous row widget is deleted and the per-mineral widget
        dicts cleared first, so a `deleteLater`-ed row can never
        deliver a late focus or change event into a stale dict
        entry (the handlers guard on `_rebuilding` as a second
        line of defence).  Called on startup and by
        `_after_draft_change` after every draft change; it holds
        the `_rebuilding` guard itself, so populating the new rows
        commits nothing.
        """
        self._rebuilding = True
        try:
            layout: QVBoxLayout = self._kinetics_layout
            while layout.count() > 0:
                item: QLayoutItem | None = layout.takeAt(0)
                if item is None:
                    continue
                widget: QWidget | None = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._kinetics_type.clear()
            self._kinetics_label.clear()
            self._kinetics_k.clear()
            self._kinetics_terms.clear()
            for mineral in self._draft.mineral:
                layout.addWidget(self._build_kinetics_row(mineral))
        finally:
            self._rebuilding = False

    def _build_kinetics_row(self, mineral: DraftMineral) -> QWidget:
        """
        One kinetics row for `mineral`: its name, a TST/Monod type
        combo, the reaction-label field, the log10 k spin, and the
        term table.

        Every widget is built fully before any signal is connected,
        and the caller (`_rebuild_kinetics_rows`) holds the
        `_rebuilding` guard, so populating the row commits
        nothing.  The row's widgets are indexed in the per-mineral
        dicts by name for the commit handlers; the entry's kind
        (TST vs Monod) comes from which of the draft's two kinetics
        sections carries it, with a kinetics-less mineral defaulting
        to TST.
        """
        entry: DraftTst | DraftMonod | None = self._kinetics_entry(
            mineral.name
        )
        kind: str = "Monod" if isinstance(entry, DraftMonod) else "TST"
        container: QWidget = QWidget()
        grid: QGridLayout = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        name_label: QLabel = QLabel(mineral.name)
        name_label.setStyleSheet("font-weight: bold;")
        grid.addWidget(name_label, 0, 0)
        grid.addWidget(QLabel("Type:"), 0, 1)
        type_combo: QComboBox = QComboBox()
        type_combo.addItems(["TST", "Monod"])
        type_combo.setCurrentText(kind)
        grid.addWidget(type_combo, 0, 2)
        grid.addWidget(QLabel("Reaction:"), 1, 1)
        label_field: QWidget = self._build_label_field(
            mineral.name,
            kind,
            entry.label if entry is not None else "",
        )
        grid.addWidget(label_field, 1, 2)
        grid.addWidget(QLabel("log10 k:"), 2, 1)
        k_spin: QDoubleSpinBox = _make_spin(-30.0, 30.0, 6)
        k_spin.setValue(entry.log10_k if entry is not None else 0.0)
        grid.addWidget(k_spin, 2, 2)
        table: QTableWidget = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Kind", "Species", "Coeff."])
        table.setColumnWidth(1, 160)
        offered: list[str] = sorted(
            _draft_carried(self._draft) | _kinetics_term_keys(entry)
        )
        self._fill_terms_table(
            table, mineral.name, kind, entry, offered
        )
        grid.addWidget(table, 3, 0, 1, 3)
        type_combo.currentIndexChanged.connect(
            lambda _index, n=mineral.name, combo=type_combo: (
                self._on_kinetics_type_changed(n, combo)
            )
        )
        k_spin.valueChanged.connect(
            lambda _value, n=mineral.name, spin=k_spin: (
                self._on_kinetics_k_changed(n, spin)
            )
        )
        self._kinetics_type[mineral.name] = type_combo
        self._kinetics_label[mineral.name] = label_field
        self._kinetics_k[mineral.name] = k_spin
        self._kinetics_terms[mineral.name] = table
        return container

    def _build_label_field(
        self, name: str, kind: str, current_label: str
    ) -> QWidget:
        """
        The reaction-label field for the mineral `name` of `kind`.

        A database offering several labels for the mineral/kind
        gets a combo (picking a label re-seeds log10 k and the
        terms from the database, `_on_kinetics_label_changed`);
        one label gets a line edit prefilled with it; none (a
        draft-only mineral) gets a free-form line edit, and a
        blank committed label is V6's to flag.  Built before
        connection; the `_rebuilding` guard applies.
        """
        if kind == "Monod":
            db_labels: list[str] = sorted(
                self._database.monod_reactions.get(name, {})
            )
        else:
            db_labels = sorted(
                self._database.tst_reactions.get(name, {})
            )
        if len(db_labels) > 1:
            combo: QComboBox = QComboBox()
            combo.addItems(db_labels)
            if current_label in db_labels:
                combo.setCurrentText(current_label)
            combo.currentIndexChanged.connect(
                lambda _index, n=name, field=combo: (
                    self._on_kinetics_label_changed(n, field)
                )
            )
            return combo
        edit: QLineEdit = QLineEdit()
        if db_labels:
            edit.setText(db_labels[0])
        elif current_label:
            edit.setText(current_label)
        edit.returnPressed.connect(
            lambda n=name, field=edit: self._on_kinetics_label_changed(
                n, field
            )
        )
        def _commit_label(n: str = name, field: QWidget = edit) -> None:
            self._on_kinetics_label_changed(n, field)

        edit.installEventFilter(_FocusOutFilter(edit, _commit_label))
        return edit

    def _fill_terms_table(
        self,
        table: QTableWidget,
        name: str,
        kind: str,
        entry: DraftTst | DraftMonod | None,
        offered: list[str],
    ) -> None:
        """
        Fill the term table of the row of the mineral `name`.

        TST entries show one row per dependence term; Monod
        entries show their monod terms then their inhibitation
        terms, each row tagged by kind in a non-editable item.
        The species combos offer `offered` (the draft's carried
        set plus the entry's own keys), so a dangling reference
        stays selectable — the V7 finding must remain visible
        instead of being normalized away.  `QTableWidget`'s
        cell-changed signal does not fire for cell *widgets*, so
        every combo and spin gets its own connection to
        `_on_kinetics_term_changed`.  Called under the
        `_rebuilding` guard.
        """
        rows: list[tuple[str, str, float]] = []
        if entry is not None:
            if isinstance(entry, DraftTst):
                rows = [
                    ("TST", key, value)
                    for key, value in sorted(entry.dependence)
                ]
            else:
                rows = [
                    ("Monod", key, value)
                    for key, value in sorted(entry.monod_terms)
                ] + [
                    ("Inhib", key, value)
                    for key, value in sorted(entry.inhib_terms)
                ]
        for index, (row_kind, key, value) in enumerate(rows):
            table.insertRow(index)
            kind_item: QTableWidgetItem = QTableWidgetItem(row_kind)
            kind_item.setFlags(
                kind_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            table.setItem(index, 0, kind_item)
            combo: QComboBox = QComboBox()
            combo.addItems(offered)
            if key in offered:
                combo.setCurrentText(key)
            combo.currentIndexChanged.connect(
                lambda _index, n=name, t=table: (
                    self._on_kinetics_term_changed(n, t)
                )
            )
            spin: QDoubleSpinBox = _make_spin(-1000.0, 1000.0, 6)
            spin.setValue(value)
            spin.valueChanged.connect(
                lambda _value, n=name, t=table: (
                    self._on_kinetics_term_changed(n, t)
                )
            )
            table.setCellWidget(index, 1, combo)
            table.setCellWidget(index, 2, spin)

    def _kinetics_entry(
        self, name: str
    ) -> DraftTst | DraftMonod | None:
        """
        The draft's kinetics entry for the mineral `name`, if any.

        A mineral may legally have neither (V5 flags it); the
        handlers treat that as "no label, zero k, no terms" and
        create the entry on the first committed edit.
        """
        tst: DraftTst | None = next(
            (t for t in self._draft.tst if t.mineral_name == name),
            None,
        )
        if tst is not None:
            return tst
        return next(
            (m for m in self._draft.monod if m.mineral_name == name),
            None,
        )

    def _upsert_kinetics(
        self,
        name: str,
        kind: str,
        label: str,
        log10_k: float,
        dependence: Sequence[tuple[str, float]] = (),
        monod_terms: Sequence[tuple[str, float]] = (),
        inhib_terms: Sequence[tuple[str, float]] = (),
    ) -> None:
        """
        Upsert the mineral's single kinetics entry of `kind`.

        `upsert_tst` / `upsert_monod` each drop the other kind for
        the same mineral, so the upsert enforces V5 (exactly one
        entry per mineral) on every edit.
        """
        if kind == "Monod":
            self._draft = self._draft.upsert_monod(
                DraftMonod(
                    name,
                    label,
                    log10_k,
                    tuple(monod_terms),
                    tuple(inhib_terms),
                )
            )
        else:
            self._draft = self._draft.upsert_tst(
                DraftTst(name, label, log10_k, tuple(dependence))
            )

    def _seeded_kinetic(
        self, name: str, kind: str, label: str
    ) -> _SeededKinetic | None:
        """
        The database's kinetics reaction for (mineral, kind,
        label), or None.

        Returns `(log10 k, dependence, monod terms, inhib terms)`
        with the term pairs sorted by species key; only the list
        matching `kind` is populated, the other two are empty.
        """
        if kind == "Monod":
            monod_map: dict[str, pt.MonodReaction] = (
                self._database.monod_reactions.get(name) or {}
            )
            monod: pt.MonodReaction | None = monod_map.get(label)
            if monod is None:
                return None
            return (
                monod.rate_constant,
                [],
                [
                    (key, value)
                    for key, value in sorted(monod.monod_terms.items())
                ],
                [
                    (key, value)
                    for key, value in sorted(monod.inhib_terms.items())
                ],
            )
        tst_map: dict[str, pt.TstReaction] = (
            self._database.tst_reactions.get(name) or {}
        )
        tst: pt.TstReaction | None = tst_map.get(label)
        if tst is None:
            return None
        return (
            tst.rate_constant,
            [
                (key, value)
                for key, value in sorted(tst.dependence.items())
            ],
            [],
            [],
        )

    def _on_kinetics_type_changed(
        self, name: str, type_combo: QComboBox
    ) -> None:
        """
        The mineral's kinetics type changed (TST ↔ Monod).

        The entry is re-created as the new kind: a database
        reaction for (mineral, new kind, the entry's current
        label) seeds log10 k and the terms; without one, log10 k
        is kept and the terms start empty (the old kind's terms do
        not translate across).  The label field's shape depends on
        the kind (which labels the database offers), so the whole
        row rebuilds.
        """
        if self._rebuilding or name not in self._kinetics_type:
            return
        kind: str = type_combo.currentText()
        entry: DraftTst | DraftMonod | None = self._kinetics_entry(name)
        label: str = entry.label if entry is not None else ""
        log10_k: float = entry.log10_k if entry is not None else 0.0
        seeded: _SeededKinetic | None = self._seeded_kinetic(
            name, kind, label
        )
        if seeded is not None:
            log10_k = seeded[0]
        self._upsert_kinetics(
            name,
            kind,
            label,
            log10_k,
            dependence=seeded[1] if seeded is not None else (),
            monod_terms=seeded[2] if seeded is not None else (),
            inhib_terms=seeded[3] if seeded is not None else (),
        )
        self._dirty = True
        self._rebuild_kinetics_rows()
        self._refresh_validation()

    def _on_kinetics_label_changed(self, name: str, field: QWidget) -> None:
        """
        The mineral's reaction label committed (combo pick, or
        line edit return/focus-out).

        A database reaction for (mineral, current kind, new
        label) re-seeds log10 k and the terms and rebuilds the
        row — without the rebuild, a later term commit would
        overwrite the seed with the stale table.  Otherwise only
        the label changes: the entry keeps its log10 k and its
        terms.
        """
        if self._rebuilding or name not in self._kinetics_type:
            return
        if isinstance(field, QComboBox):
            label: str = field.currentText()
        else:
            assert isinstance(field, QLineEdit)
            label = field.text().strip()
        kind: str = self._kinetics_type[name].currentText()
        entry: DraftTst | DraftMonod | None = self._kinetics_entry(name)
        seeded: _SeededKinetic | None = self._seeded_kinetic(
            name, kind, label
        )
        if seeded is not None:
            self._upsert_kinetics(
                name,
                kind,
                label,
                seeded[0],
                dependence=seeded[1],
                monod_terms=seeded[2],
                inhib_terms=seeded[3],
            )
            self._dirty = True
            self._rebuild_kinetics_rows()
            self._refresh_validation()
            return
        log10_k: float = entry.log10_k if entry is not None else 0.0
        if entry is None:
            self._upsert_kinetics(name, kind, label, log10_k)
        elif isinstance(entry, DraftTst):
            self._upsert_kinetics(
                name,
                kind,
                label,
                log10_k,
                dependence=list(entry.dependence),
            )
        else:
            self._upsert_kinetics(
                name,
                kind,
                label,
                log10_k,
                monod_terms=list(entry.monod_terms),
                inhib_terms=list(entry.inhib_terms),
            )
        self._dirty = True
        self._refresh_validation()

    def _on_kinetics_k_changed(
        self, name: str, k_spin: QDoubleSpinBox
    ) -> None:
        """
        The mineral's log10 k changed.

        The entry (or a bare one, if the mineral has none yet) is
        re-upserted with the new constant, its label and terms
        intact.  The row is deliberately not rebuilt — a spin box
        mid-typing must not be re-rendered under the user.
        """
        if self._rebuilding or name not in self._kinetics_type:
            return
        entry: DraftTst | DraftMonod | None = self._kinetics_entry(name)
        kind: str = self._kinetics_type[name].currentText()
        label: str = entry.label if entry is not None else ""
        if entry is None:
            self._upsert_kinetics(name, kind, label, k_spin.value())
        elif isinstance(entry, DraftTst):
            self._upsert_kinetics(
                name,
                kind,
                label,
                k_spin.value(),
                dependence=list(entry.dependence),
            )
        else:
            self._upsert_kinetics(
                name,
                kind,
                label,
                k_spin.value(),
                monod_terms=list(entry.monod_terms),
                inhib_terms=list(entry.inhib_terms),
            )
        self._dirty = True
        self._refresh_validation()

    def _on_kinetics_term_changed(
        self, name: str, table: QTableWidget
    ) -> None:
        """
        A kinetics term changed (species picked, coefficient
        edited): commit the whole table.

        The rows are grouped by their kind column back into the
        dependence / monod / inhib lists the entry carries; the
        table is left as the user set it (no rebuild).
        """
        if self._rebuilding or name not in self._kinetics_type:
            return
        entry: DraftTst | DraftMonod | None = self._kinetics_entry(name)
        kind: str = self._kinetics_type[name].currentText()
        label: str = entry.label if entry is not None else ""
        log10_k: float = entry.log10_k if entry is not None else 0.0
        dependence: list[tuple[str, float]] = []
        monod_terms: list[tuple[str, float]] = []
        inhib_terms: list[tuple[str, float]] = []
        for index in range(table.rowCount()):
            kind_item: QTableWidgetItem | None = table.item(index, 0)
            combo: QWidget | None = table.cellWidget(index, 1)
            spin: QWidget | None = table.cellWidget(index, 2)
            if kind_item is None or combo is None or spin is None:
                continue
            assert isinstance(combo, QComboBox)
            assert isinstance(spin, QDoubleSpinBox)
            pair: tuple[str, float] = (
                combo.currentText(),
                spin.value(),
            )
            if kind_item.text() == "TST":
                dependence.append(pair)
            elif kind_item.text() == "Monod":
                monod_terms.append(pair)
            else:
                inhib_terms.append(pair)
        self._upsert_kinetics(
            name,
            kind,
            label,
            log10_k,
            dependence=dependence,
            monod_terms=monod_terms,
            inhib_terms=inhib_terms,
        )
        self._dirty = True
        self._refresh_validation()

    def _on_add_from_database(self) -> None:
        """
        The "+ Add from database" button: insert a database entry
        into the current (or a picked) section.

        The section picker opens only when nothing is selected;
        the entry picker lists the database's names for the
        section that the draft does not already carry.  The picked
        entry is selected in the browsers and detail pane
        afterwards (`_upsert_db_entry` ends in
        `_after_draft_change`, which keeps a valid selection).
        """
        titles: list[str] = [_SECTION_TITLES[s] for s in _SECTIONS]
        if self._selected is not None:
            section: str = self._selected[0]
        else:
            title, picked = QInputDialog.getItem(
                self,
                "Add from database",
                "Section:",
                titles,
                0,
                False,
            )
            if not picked:
                return
            section = _SECTIONS[titles.index(title)]
        candidates: list[str] = sorted(
            name
            for name in self._db_names(section)
            if name not in self._draft_names(section)
        )
        if not candidates:
            QMessageBox.information(
                self,
                "Add from database",
                "Every database entry is already in the draft.",
            )
            return
        name, picked = QInputDialog.getItem(
            self,
            "Add from database",
            f"{_SECTION_TITLES[section]} entry:",
            candidates,
            0,
            False,
        )
        if not picked:
            return
        self._selected = (section, name)
        self._upsert_db_entry(section, name)

    def _on_new_entry(self) -> None:
        """
        The "New…" button: create a blank draft-only entry in the
        current (or a picked) section and select it.

        The entry starts with zero parameters; a mineral also
        gets a blank TST kinetics entry (the type combo in its
        row is the user's to flip to Monod).  Validation flags
        what is missing: V6 (empty reaction label) for the new
        mineral, V4 (missing self-term) for a new secondary or
        exchange.  A blank name, or a name the section already
        carries, is refused.
        """
        titles: list[str] = [_SECTION_TITLES[s] for s in _SECTIONS]
        if self._selected is not None:
            section: str = self._selected[0]
        else:
            title, picked = QInputDialog.getItem(
                self,
                "New entry",
                "Section:",
                titles,
                0,
                False,
            )
            if not picked:
                return
            section = _SECTIONS[titles.index(title)]
        name, picked = QInputDialog.getText(
            self,
            "New entry",
            f"{_SECTION_TITLES[section]} name:",
        )
        name = name.strip()
        if not picked or not name:
            return
        if name in self._draft_names(section):
            QMessageBox.critical(
                self,
                "New entry",
                f"{name} is already in the "
                f"{_SECTION_TITLES[section].lower()} section.",
            )
            return
        if section == "primary":
            self._draft = self._draft.upsert_primary(
                DraftPrimary(name, 0.0, 0.0, 0.0)
            )
        elif section == "secondary":
            self._draft = self._draft.upsert_secondary(
                DraftSecondary(name, (), (0.0, 0.0), 0.0, 0.0, 0.0)
            )
        elif section == "mineral":
            self._draft = self._draft.upsert_mineral(
                DraftMineral(name, 0.0, (), (0.0, 0.0), 0.0)
            )
            self._draft = self._draft.upsert_tst(
                DraftTst(name, "", 0.0, ())
            )
        else:
            self._draft = self._draft.upsert_exchange(
                DraftExchange(name, (), 0.0, 0.0)
            )
        self._selected = (section, name)
        self._after_draft_change()


def edit_network(
    network: pt.ReactionNetwork,
    database: pt.ChemicalDatabase,
    parent: QWidget | None = None,
) -> pt.ReactionNetwork | None:
    """
    Open the network editor and ask for a replacement network.

    The dialog runs modally; its result is read from the `network`
    property, mirroring `ask_calibration_settings`.

    Args:
        network: The network the editor starts from (a preset, a
            file load, or a previously applied draft).
        database: The default chemical database for the browser
            and for "+ Add from database".
        parent: The dialog's parent widget; defaults to None.

    Returns:
        The edited network, or None when the dialog was cancelled.
    """
    dialog: NetworkEditorDialog = NetworkEditorDialog(
        network, database, parent
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.network
