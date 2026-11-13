"""Pandas dataframe views over the compiled ``ChemicalDatabase``.

The Rust database stores its contents as mappings keyed by name, which is
efficient for lookup but awkward for exploration.  These functions turn each
collection into a pandas ``DataFrame`` so the contents can be filtered,
joined and visualized with the usual pandas idioms.

Two layouts are offered:

* *wide* -- one row per entry (species or reaction), scalar attributes as
  columns.  The variable-length ``eq_consts`` vector (8 entries in the
  default database) is kept as a list-valued column.  ``log10_k_eq`` is a
  convenience column: for exchange reactions it is the stored constant, and
  for secondary / mineral species it is ``eq_consts[1]``, the value the
  speciation solve actually consumes.
* *long* -- one row per ``(entry, component)`` stoichiometry pair, which
  keeps the variable-length stoichiometry dictionaries rectangular and makes
  questions like "which species contain uranium?" a single filter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd

from ..core import (
    ChemicalDatabase,
    ExchangeReaction,
    MineralSpecies,
    MonodReaction,
    PrimaryAqueousSpecies,
    SecondarySpecies,
    TstReaction,
)


class StoichiometricEntry(Protocol):
    """An entry that has a name and a variable-length stoichiometry mapping."""

    name: str
    stoichiometry: dict[str, float]


def log10_k(eq_consts: list[float]) -> float:
    """
    Return the log10-K value used by the speciation solve.

    The database stores a vector of constants per species, but the
    equilibrium machinery only consumes the entry at index 1.  Return that
    value, or NaN when the vector is too short to contain it.

    Args:
        eq_consts:
            The stored constant vector for one species.

    Returns:
        The consumed constant, or NaN when unavailable.
    """
    if len(eq_consts) > 1:
        return eq_consts[1]
    return float("nan")


def stoich_rows(entries: Sequence[StoichiometricEntry]) -> list[tuple[str, str, float]]:
    """
    Melt the stoichiometry mappings of the entries into long-format rows.

    Args:
        entries:
            The database entries to melt, in the desired row order.

    Returns:
        One ``(name, component, coefficient)`` row per non-zero
        stoichiometry entry, with components in sorted order within each
        entry.
    """
    return [
        (entry.name, component, coefficient)
        for entry in entries
        for component, coefficient in sorted(entry.stoichiometry.items())
    ]


def primary_species_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a wide dataframe of the primary aqueous species.

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per primary species, sorted by name, with the species name
        as the index and the columns ``molar_mass``, ``charge`` and
        ``dh_size_param``.
    """
    species: list[PrimaryAqueousSpecies] = db.get_primary_aqueous_species(
        db.get_primary_aqueous_species_names()
    )
    return pd.DataFrame(
        {
            "molar_mass": [s.molar_mass for s in species],
            "charge": [s.charge for s in species],
            "dh_size_param": [s.dh_size_param for s in species],
        },
        index=pd.Index([s.name for s in species], name="name"),
    )


def secondary_species_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a wide dataframe of the secondary (equilibrium) species.

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per secondary species, sorted by name, with the species name
        as the index and the columns ``molar_mass``, ``charge``,
        ``dh_size_param``, ``log10_k_eq`` and ``eq_consts``.  ``log10_k_eq``
        is ``eq_consts[1]`` (the value consumed by the speciation solve);
        ``eq_consts`` is kept as a list-valued column because its remaining
        entries have no documented per-component meaning.
    """
    species: list[SecondarySpecies] = db.get_secondary_species(
        db.get_secondary_species_names()
    )
    eq_consts: list[list[float]] = [s.eq_consts for s in species]
    return pd.DataFrame(
        {
            "molar_mass": [s.molar_mass for s in species],
            "charge": [s.charge for s in species],
            "dh_size_param": [s.dh_size_param for s in species],
            "log10_k_eq": [log10_k(c) for c in eq_consts],
            "eq_consts": eq_consts,
        },
        index=pd.Index([s.name for s in species], name="name"),
    )


def secondary_stoich_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a long dataframe of the secondary-species stoichiometries.

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per ``(species, component)`` stoichiometry pair, sorted by
        species then component, with the columns ``species``, ``component``
        and ``stoich``.
    """
    species: list[SecondarySpecies] = db.get_secondary_species(
        db.get_secondary_species_names()
    )
    return pd.DataFrame(
        stoich_rows(species), columns=["species", "component", "stoich"]
    )


def mineral_species_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a wide dataframe of the mineral species.

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per mineral, sorted by name, with the mineral name as the
        index and the columns ``molar_mass``, ``molar_volume``,
        ``log10_k_eq`` and ``eq_consts``.  ``log10_k_eq`` is ``eq_consts[1]``
        (the value consumed by the speciation solve); ``eq_consts`` is kept
        as a list-valued column because its remaining entries have no
        documented per-component meaning.
    """
    species: list[MineralSpecies] = db.get_mineral_species(
        db.get_mineral_species_names()
    )
    eq_consts: list[list[float]] = [s.eq_consts for s in species]
    return pd.DataFrame(
        {
            "molar_mass": [s.molar_mass for s in species],
            "molar_volume": [s.molar_volume for s in species],
            "log10_k_eq": [log10_k(c) for c in eq_consts],
            "eq_consts": eq_consts,
        },
        index=pd.Index([s.name for s in species], name="name"),
    )


def mineral_stoich_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a long dataframe of the mineral stoichiometries.

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per ``(mineral, component)`` stoichiometry pair, sorted by
        mineral then component, with the columns ``mineral``, ``component``
        and ``stoich``.
    """
    species: list[MineralSpecies] = db.get_mineral_species(
        db.get_mineral_species_names()
    )
    return pd.DataFrame(
        stoich_rows(species), columns=["mineral", "component", "stoich"]
    )


def exchange_reactions_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a wide dataframe of the ion-exchange reactions.

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per exchange species, sorted by name, with the species name
        as the index and the columns ``log10_k_eq`` and ``charge``.
    """
    reactions: list[ExchangeReaction] = db.get_exchange_reactions(
        db.get_exchange_species_names()
    )
    return pd.DataFrame(
        {
            "log10_k_eq": [r.log10_k_eq for r in reactions],
            "charge": [r.charge for r in reactions],
        },
        index=pd.Index([r.name for r in reactions], name="name"),
    )


def exchange_stoich_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a long dataframe of the exchange-reaction stoichiometries.

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per ``(exchange_species, component)`` stoichiometry pair,
        sorted by exchange species then component, with the columns
        ``exchange_species``, ``component`` and ``stoich``.
    """
    reactions: list[ExchangeReaction] = db.get_exchange_reactions(
        db.get_exchange_species_names()
    )
    return pd.DataFrame(
        stoich_rows(reactions),
        columns=["exchange_species", "component", "stoich"],
    )


def tst_reactions_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a wide dataframe of the TST (transition-state) kinetic reactions.

    Each (mineral, label) reaction becomes one row.  The reaction-order
    dependence is spread across one column per dependent species (the union
    over all reactions), NaN where a reaction does not depend on that
    species.

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per (mineral, label), indexed by the multi-index
        ``(mineral_name, label)`` in sorted order, with the column
        ``rate_constant`` plus one column per dependent species.
    """
    fixed_columns = ["mineral_name", "label", "rate_constant"]
    rows: list[dict[str, float | str]] = []
    for mineral_name in db.get_tst_mineral_names():
        for label, reaction in sorted(db.tst_reactions[mineral_name].items()):
            row: dict[str, float | str] = {
                "mineral_name": mineral_name,
                "label": label,
                "rate_constant": reaction.rate_constant,
            }
            row.update(reaction.dependence)
            rows.append(row)
    extra_columns = sorted({key for row in rows for key in row} - set(fixed_columns))
    frame = pd.DataFrame(rows, columns=fixed_columns + extra_columns)
    return frame.set_index(["mineral_name", "label"])


def monod_reactions_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a wide dataframe of the Monod (organic-matter) kinetic reactions.

    Each (mineral, label) reaction becomes one row.  The substrate and
    inhibitor terms are spread across one column per species, prefixed with
    ``monod_`` and ``inhib_`` respectively to keep the two term types
    distinct, NaN where a reaction does not carry that term.

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per (mineral, label), indexed by the multi-index
        ``(mineral_name, label)`` in sorted order, with the column
        ``rate_constant`` plus ``monod_<species>`` and ``inhib_<species>``
        columns for the union of the term species.
    """
    fixed_columns = ["mineral_name", "label", "rate_constant"]
    rows: list[dict[str, float | str]] = []
    for mineral_name in db.get_monod_mineral_names():
        for label, reaction in sorted(db.monod_reactions[mineral_name].items()):
            row: dict[str, float | str] = {
                "mineral_name": mineral_name,
                "label": label,
                "rate_constant": reaction.rate_constant,
            }
            row.update({f"monod_{key}": value for key, value in reaction.monod_terms.items()})
            row.update({f"inhib_{key}": value for key, value in reaction.inhib_terms.items()})
            rows.append(row)
    extra_columns = sorted({key for row in rows for key in row} - set(fixed_columns))
    frame = pd.DataFrame(rows, columns=fixed_columns + extra_columns)
    return frame.set_index(["mineral_name", "label"])


def all_species_df(db: ChemicalDatabase) -> pd.DataFrame:
    """
    Build a quick-look dataframe combining all four species types.

    Intended for orienting oneself in the database: the per-type frames are
    the full-fidelity views, this one is a compact union (the list-valued
    ``eq_consts`` column is omitted here in favor of ``log10_k_eq``).

    Args:
        db:
            The chemical database to view.

    Returns:
        One row per species across all four types, with the columns ``kind``
        (``primary`` / ``secondary`` / ``mineral`` / ``exchange``), ``name``,
        ``molar_mass``, ``charge``, ``dh_size_param``, ``molar_volume`` and
        ``log10_k_eq``, NaN where an attribute does not apply to the type.
    """
    parts: list[pd.DataFrame] = [
        primary_species_df(db).reset_index().assign(kind="primary"),
        secondary_species_df(db)
        .reset_index()
        .assign(kind="secondary")
        .drop(columns=["eq_consts"]),
        mineral_species_df(db)
        .reset_index()
        .assign(kind="mineral")
        .drop(columns=["eq_consts"]),
        exchange_reactions_df(db).reset_index().assign(kind="exchange"),
    ]
    columns = [
        "kind",
        "name",
        "molar_mass",
        "charge",
        "dh_size_param",
        "molar_volume",
        "log10_k_eq",
    ]
    return pd.concat(parts, ignore_index=True)[columns]
