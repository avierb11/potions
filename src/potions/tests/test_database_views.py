"""Tests for the pandas dataframe views over the compiled ChemicalDatabase.

Covers the row / column layout of each wide and long frame, spot values
against the checked-in default database, the derived ``log10_k_eq``
convention, and the combined ``all_species_df`` union.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..core import ChemicalDatabase
from ..reactive_transport.database_views import (
    all_species_df,
    exchange_reactions_df,
    exchange_stoich_df,
    mineral_species_df,
    mineral_stoich_df,
    monod_reactions_df,
    primary_species_df,
    secondary_species_df,
    secondary_stoich_df,
    tst_reactions_df,
)


@pytest.fixture(scope="module")
def db() -> ChemicalDatabase:
    return ChemicalDatabase.load_default()


# ===================== Wide species frames ===================== #
def test_primary_species_df(db: ChemicalDatabase) -> None:
    frame = primary_species_df(db)
    assert frame.shape == (100, 3)
    assert list(frame.columns) == ["molar_mass", "charge", "dh_size_param"]
    assert frame.index.name == "name"
    assert frame.index.is_unique
    assert frame.index.is_monotonic_increasing

    doc = frame.loc["DOC"]
    assert doc["molar_mass"] == 300.0
    assert doc["charge"] == 0.0
    assert doc["dh_size_param"] == 4.0


def test_secondary_species_df(db: ChemicalDatabase) -> None:
    frame = secondary_species_df(db)
    assert frame.shape == (1699, 5)
    assert list(frame.columns) == [
        "molar_mass",
        "charge",
        "dh_size_param",
        "log10_k_eq",
        "eq_consts",
    ]
    assert frame.index.name == "name"
    assert frame.index.is_unique
    assert frame.index.is_monotonic_increasing

    co2 = frame.loc["CO2(aq)"]
    assert co2["molar_mass"] == 44.0098
    assert co2["charge"] == 0.0
    assert co2["dh_size_param"] == 3.0
    assert isinstance(co2["eq_consts"], list)
    assert len(co2["eq_consts"]) == 8


def test_mineral_species_df(db: ChemicalDatabase) -> None:
    frame = mineral_species_df(db)
    assert frame.shape == (1153, 4)
    assert list(frame.columns) == [
        "molar_mass",
        "molar_volume",
        "log10_k_eq",
        "eq_consts",
    ]
    assert frame.index.name == "name"
    assert frame.index.is_unique
    assert frame.index.is_monotonic_increasing

    soc = frame.loc["SOC(s)"]
    assert soc["molar_mass"] == 180.0
    assert soc["molar_volume"] == 115.48
    assert soc["log10_k_eq"] == -3.0


def test_exchange_reactions_df(db: ChemicalDatabase) -> None:
    frame = exchange_reactions_df(db)
    assert frame.shape == (33, 2)
    assert list(frame.columns) == ["log10_k_eq", "charge"]
    assert frame.index.name == "name"
    assert frame.index.is_unique
    assert frame.index.is_monotonic_increasing

    xdoc = frame.loc["XDOC"]
    assert xdoc["log10_k_eq"] == -2.75
    assert xdoc["charge"] == 0.0


# ===================== Long stoichiometry frames ===================== #
def test_secondary_stoich_df(db: ChemicalDatabase) -> None:
    frame = secondary_stoich_df(db)
    assert frame.shape == (6252, 3)
    assert list(frame.columns) == ["species", "component", "stoich"]

    co2 = frame[frame["species"] == "CO2(aq)"]
    pairs = {
        (row.component, row.stoich)
        for row in co2.itertuples(index=False)
    }
    assert pairs == {
        ("CO2(aq)", -1.0),
        ("H+", 1.0),
        ("H2O", -1.0),
        ("HCO3-", 1.0),
    }


def test_mineral_stoich_df(db: ChemicalDatabase) -> None:
    frame = mineral_stoich_df(db)
    assert frame.shape == (5228, 3)
    assert list(frame.columns) == ["mineral", "component", "stoich"]

    soc = frame[frame["mineral"] == "SOC(s)"]
    pairs = {(row.component, row.stoich) for row in soc.itertuples(index=False)}
    assert pairs == {
        ("CO2(aq)", 0.6),
        ("DOC", 0.55),
        ("SOC(s)", -1.0),
    }


def test_exchange_stoich_df(db: ChemicalDatabase) -> None:
    frame = exchange_stoich_df(db)
    assert frame.shape == (99, 3)
    assert list(frame.columns) == ["exchange_species", "component", "stoich"]

    xdoc = frame[frame["exchange_species"] == "XDOC"]
    pairs = {(row.component, row.stoich) for row in xdoc.itertuples(index=False)}
    assert pairs == {
        ("DOC", 1.0),
        ("X-", 1.0),
        ("XDOC", -1.0),
    }


# ===================== Kinetic reaction frames ===================== #
def test_tst_reactions_df(db: ChemicalDatabase) -> None:
    frame = tst_reactions_df(db)
    assert frame.shape == (86, 6)
    assert frame.index.names == ["mineral_name", "label"]
    assert frame.index.is_unique
    assert frame.index.is_monotonic_increasing
    # The fixed column plus one column per dependent species.
    assert list(frame.columns) == [
        "rate_constant",
        "Fe++",
        "Fe+++",
        "H+",
        "O2(aq)",
        "OH-",
    ]

    flat = frame.reset_index()
    gypsum = flat[(flat["mineral_name"] == "Gypsum") & (flat["label"] == "default")]
    # Six data columns plus the two index columns restored by reset_index().
    assert gypsum.shape == (1, 8)
    assert gypsum["rate_constant"].iloc[0] == -6.0
    # Gypsum carries no dependence terms, so every term column is NaN.
    assert gypsum[["Fe++", "Fe+++", "H+", "O2(aq)", "OH-"]].isna().all().all()


def test_monod_reactions_df(db: ChemicalDatabase) -> None:
    frame = monod_reactions_df(db)
    assert frame.shape == (10, 7)
    assert frame.index.names == ["mineral_name", "label"]
    assert frame.index.is_unique
    assert frame.index.is_monotonic_increasing
    # The default database carries no inhibitor terms, so no inhib_ columns.
    assert list(frame.columns) == [
        "rate_constant",
        "monod_DOC",
        "monod_NO3-",
        "monod_O2(aq)",
        "monod_SO4--",
        "monod_SOC",
        "monod_SOC(s)",
    ]

    flat = frame.reset_index()
    soc = flat[(flat["mineral_name"] == "SOC(s)") & (flat["label"] == "test")]
    # Seven data columns plus the two index columns restored by reset_index().
    assert soc.shape == (1, 9)
    assert soc["rate_constant"].iloc[0] == -10.0
    assert soc["monod_SOC(s)"].iloc[0] == pytest.approx(6e-6)
    assert soc[
        ["monod_DOC", "monod_NO3-", "monod_O2(aq)", "monod_SO4--", "monod_SOC"]
    ].isna().all().all()


# ===================== Combined view ===================== #
def test_all_species_df(db: ChemicalDatabase) -> None:
    frame = all_species_df(db)
    assert frame.shape == (2985, 7)
    assert list(frame.columns) == [
        "kind",
        "name",
        "molar_mass",
        "charge",
        "dh_size_param",
        "molar_volume",
        "log10_k_eq",
    ]
    assert isinstance(frame.index, pd.RangeIndex)

    kind_counts = frame["kind"].value_counts()
    assert kind_counts["primary"] == 100
    assert kind_counts["secondary"] == 1699
    assert kind_counts["mineral"] == 1153
    assert kind_counts["exchange"] == 33

    # Inapplicable attributes are NaN: minerals carry no charge, exchange
    # species carry no molar volume.
    assert frame.loc[frame["kind"] == "mineral", "charge"].isna().all()
    assert frame.loc[frame["kind"] == "exchange", "molar_volume"].isna().all()
    # Primary species carry neither a log10_k_eq nor a molar volume.
    assert frame.loc[frame["kind"] == "primary", "log10_k_eq"].isna().all()


def test_log10_k_eq_matches_eq_consts(db: ChemicalDatabase) -> None:
    # The derived column must agree with eq_consts[1] in every row of both
    # frames that keep the raw vector.
    for frame in (secondary_species_df(db), mineral_species_df(db)):
        raw = pd.Series(
            [
                consts[1] if len(consts) > 1 else float("nan")
                for consts in frame["eq_consts"]
            ],
            index=frame.index,
        )
        assert np.allclose(frame["log10_k_eq"], raw, equal_nan=True)
