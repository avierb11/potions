"""Regression tests for the TST parameter construction.

Both ``tst_params`` implementations (the Rust ``ReactionNetwork.tst_params``
getter, which is the live runtime path, and the parallel Python
``ReactionNetwork.tst_params`` property) previously suffered two defects
that turned a mineral without forward-rate dependences (e.g. calcite) into
a concentration-independent rate:

* the solubility-product exponents were gated on the reaction's (typically
  empty) dependence map, so every exponent came out zero (Q = 1), and the
  stoichiometry and dependence matrices were swapped;
* the Python property passed the raw log10 equilibrium constant, while the
  rate formula requires the linear K (``10 ** log10_K``).

The corrected construction:

* the ``stoich`` row holds the database stoichiometric exponents of the
  aqueous species; the mineral's own column is zero, because a pure phase
  has unit activity and does not enter Q;
* the ``dep`` row holds the forward-rate dependence exponents (zero where
  the reaction lists none, i.e. a factor of 1);
* ``min_eq_const`` holds the mineral equilibrium constant on the linear
  scale.

The TST rate ``D * (1 - Q/K)`` is then concentration-dependent and
self-limiting: positive (dissolution) below saturation, zero at saturation,
negative above.  A ``log10`` concentration floor keeps the rate finite for
trial states with zero or negative concentrations, which root finders probe.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ..core import (
    ChemicalDatabase as CoreDatabase,
    ReactionNetwork as CoreReactionNetwork,
    TstParameters as CoreTstParameters,
)
from ..reactive_transport.database import (
    ChemicalDatabase as PythonDatabase,
)
from ..reactive_transport.kinetic_structures import TstParameters
from ..reactive_transport.reaction_network import (
    ReactionNetwork as PythonReactionNetwork,
)

# The database stores calcite's Ksp as log10(K) = -7.4, i.e. linear
# K = 10 ** -7.4 ~= 3.981e-8.
LOG10_KSP_CALCITE: float = -7.4
KSP_CALCITE: float = 10.0 ** LOG10_KSP_CALCITE

# Species order of the minimal calcite network: primary, secondary, mineral.
MINIMAL_ORDER: list[str] = ["Ca++", "H+", "CO2(aq)", "CO3--", "Calcite(s)"]

# Species order of the full soc-calcite network used in the showcase:
# primary, secondary, minerals, then the implicit X- site and the exchange
# species.
SOC_CALCITE_ORDER: list[str] = [
    "DOC",
    "Ca++",
    "HCO3-",
    "H+",
    "CO2(aq)",
    "CO3--",
    "SOC(s)",
    "SOC-lz(s)",
    "Calcite(s)",
    "X-",
    "XDOC",
]


def _core_db() -> CoreDatabase:
    """The compiled (Rust) default chemical database."""
    return CoreDatabase.load_default()


def _minimal_calcite_network() -> CoreReactionNetwork:
    """Minimal Rust network with a single TST calcite mineral.

    Primary: Ca++, H+; secondary: CO2(aq), CO3--; no exchange species.
    """
    db: CoreDatabase = _core_db()
    return CoreReactionNetwork(
        primary_aqueous=db.get_primary_aqueous_species(["Ca++", "H+"]),
        mineral=db.get_mineral_species(["Calcite(s)"]),
        secondary=db.get_secondary_species(["CO2(aq)", "CO3--"]),
        mineral_kinetics=db.get_mineral_reactions(["Calcite(s)"], ["default"]),
        exchange_species=[],
    )


def _stoich_matrix(tst: CoreTstParameters, order: list[str]) -> np.ndarray:
    """The stoichiometry matrix restricted to (and ordered by) ``order``."""
    return tst.stoich.select(order).to_numpy()


def _dep_matrix(tst: CoreTstParameters, order: list[str]) -> np.ndarray:
    """The dependence matrix restricted to (and ordered by) ``order``."""
    return tst.dep.select(order).to_numpy()


def test_rust_stoich_matrix_is_database_exponents() -> None:
    """Q exponents: the database stoichiometric coefficients of the aqueous
    species, with the mineral's own column zero (unit activity)."""
    network: CoreReactionNetwork = _minimal_calcite_network()
    stoich: np.ndarray = _stoich_matrix(network.tst_params, MINIMAL_ORDER)

    assert stoich.shape == (1, len(MINIMAL_ORDER))
    calcite_row = stoich[0]
    assert calcite_row[MINIMAL_ORDER.index("Ca++")] == pytest.approx(1.1)
    assert calcite_row[MINIMAL_ORDER.index("CO3--")] == pytest.approx(0.5)
    assert calcite_row[MINIMAL_ORDER.index("Calcite(s)")] == 0.0
    assert calcite_row[MINIMAL_ORDER.index("H+")] == 0.0
    assert calcite_row[MINIMAL_ORDER.index("CO2(aq)")] == 0.0


def test_rust_dep_matrix_is_reaction_dependence() -> None:
    """Dependence matrix: the reaction's dependence exponents; calcite lists
    none, so every exponent is zero (forward-rate factor of 1)."""
    network: CoreReactionNetwork = _minimal_calcite_network()
    dep: np.ndarray = _dep_matrix(network.tst_params, MINIMAL_ORDER)

    assert dep.tolist() == [[0.0] * len(MINIMAL_ORDER)]


def test_rust_min_eq_const_is_linear_scale() -> None:
    """``min_eq_const`` holds the linear Ksp (``10 ** log10_K``), not the
    raw database log10 value."""
    network: CoreReactionNetwork = _minimal_calcite_network()
    k: np.ndarray = network.tst_params.min_eq_const.to_numpy()

    assert k.tolist() == pytest.approx([KSP_CALCITE])


def test_rust_rate_is_concentration_dependent_and_self_limiting() -> None:
    """``rate = D * (1 - Q/K)`` is positive and weakening below saturation,
    zero at saturation, and negative above.

    With Q = Ca++**1.1 * CO3--**0.5 and Ca++ = CO3-- = c, the saturation
    point (Q == K) sits at c = K**(1/1.6).
    """
    network: CoreReactionNetwork = _minimal_calcite_network()
    tst: CoreTstParameters = network.tst_params
    c_sat: float = KSP_CALCITE ** (1.0 / 1.6)

    def species_at(c: float) -> NDArray[np.float64]:
        vec: NDArray[np.float64] = np.zeros(len(MINIMAL_ORDER))
        vec[MINIMAL_ORDER.index("Ca++")] = c
        vec[MINIMAL_ORDER.index("CO3--")] = c
        return vec

    rate_dissolving: np.ndarray = tst.rate(species_at(1e-6))
    rate_saturated: np.ndarray = tst.rate(species_at(c_sat))
    rate_precipitating: np.ndarray = tst.rate(species_at(1e-3))

    assert rate_dissolving[0] > 0.0
    assert rate_saturated[0] == pytest.approx(0.0, abs=1e-12)
    assert rate_precipitating[0] < 0.0
    assert rate_dissolving[0] > rate_saturated[0] > rate_precipitating[0]


def test_rust_rate_finite_for_nonpositive_concentrations() -> None:
    """The ``log10`` concentration floor keeps the rate finite (no NaN) when
    a root finder probes a trial state with zero or negative
    concentrations."""
    network: CoreReactionNetwork = _minimal_calcite_network()
    tst: CoreTstParameters = network.tst_params

    bad: NDArray[np.float64] = np.zeros(len(MINIMAL_ORDER))
    bad[MINIMAL_ORDER.index("CO3--")] = -1e-9

    rates: np.ndarray = tst.rate(bad)
    assert np.isfinite(rates).all()


def test_rust_non_tst_mineral_is_zero_row() -> None:
    """A non-TST mineral (Monod here) contributes an all-zero row and a unit
    equilibrium constant, in network order, while the TST mineral's row keeps
    the database exponents."""
    db: CoreDatabase = _core_db()
    network: CoreReactionNetwork = CoreReactionNetwork(
        primary_aqueous=db.get_primary_aqueous_species(["DOC", "Ca++", "H+"]),
        mineral=db.get_mineral_species(["SOC(s)", "Calcite(s)"]),
        secondary=db.get_secondary_species(["CO2(aq)", "CO3--"]),
        mineral_kinetics=db.get_mineral_reactions(
            ["SOC(s)", "Calcite(s)"], ["test", "default"]
        ),
        exchange_species=[],
    )

    order: list[str] = [
        "DOC",
        "Ca++",
        "H+",
        "CO2(aq)",
        "CO3--",
        "SOC(s)",
        "Calcite(s)",
    ]
    tst: CoreTstParameters = network.tst_params
    stoich: np.ndarray = _stoich_matrix(tst, order)
    dep: np.ndarray = _dep_matrix(tst, order)
    k: np.ndarray = tst.min_eq_const.to_numpy()

    assert stoich.shape == (2, len(order))

    # Row 0: SOC(s) (Monod, non-TST) -> all-zero row, unit K.
    assert stoich[0].tolist() == [0.0] * len(order)
    assert dep[0].tolist() == [0.0] * len(order)
    # Row 1: calcite keeps the database exponents and the linear Ksp.
    assert stoich[1][order.index("Ca++")] == pytest.approx(1.1)
    assert stoich[1][order.index("CO3--")] == pytest.approx(0.5)
    assert stoich[1][order.index("Calcite(s)")] == 0.0
    assert k.tolist() == pytest.approx([1.0, KSP_CALCITE])


def test_rust_full_soc_calcite_network() -> None:
    """The exact showcase network: only the calcite row (the sole TST
    mineral) is non-zero; the two Monod minerals and the dependence matrix
    are all zero."""
    db: CoreDatabase = _core_db()
    network: CoreReactionNetwork = CoreReactionNetwork(
        primary_aqueous=db.get_primary_aqueous_species(
            ["DOC", "Ca++", "HCO3-", "H+"]
        ),
        mineral=db.get_mineral_species(["SOC(s)", "SOC-lz(s)", "Calcite(s)"]),
        secondary=db.get_secondary_species(["CO2(aq)", "CO3--"]),
        mineral_kinetics=db.get_mineral_reactions(
            ["SOC(s)", "SOC-lz(s)", "Calcite(s)"], ["test", "test", "default"]
        ),
        exchange_species=db.get_exchange_reactions(["XDOC"]),
    )

    assert list(network.species_order) == SOC_CALCITE_ORDER

    tst: CoreTstParameters = network.tst_params
    stoich: np.ndarray = _stoich_matrix(tst, SOC_CALCITE_ORDER)
    dep: np.ndarray = _dep_matrix(tst, SOC_CALCITE_ORDER)
    k: np.ndarray = tst.min_eq_const.to_numpy()

    assert stoich[0].tolist() == [0.0] * len(SOC_CALCITE_ORDER)
    assert stoich[1].tolist() == [0.0] * len(SOC_CALCITE_ORDER)
    assert dep.tolist() == [[0.0] * len(SOC_CALCITE_ORDER)] * 3
    assert k.tolist() == pytest.approx([1.0, 1.0, KSP_CALCITE])

    calcite_row = stoich[2]
    assert calcite_row[SOC_CALCITE_ORDER.index("Ca++")] == pytest.approx(1.1)
    assert calcite_row[SOC_CALCITE_ORDER.index("CO3--")] == pytest.approx(0.5)
    assert calcite_row[SOC_CALCITE_ORDER.index("Calcite(s)")] == 0.0


def test_python_property_matches_rust_construction() -> None:
    """The parallel Python property builds the same matrices and the
    linear-scale Ksp as the Rust getter."""
    db: PythonDatabase = PythonDatabase.load_default()
    network: PythonReactionNetwork = PythonReactionNetwork(
        primary_aqueous=db.get_primary_aqueous_species(["Ca++", "H+"]),
        mineral=db.get_mineral_species(["Calcite(s)"]),
        secondary=db.get_secondary_species(["CO2(aq)", "CO3--"]),
        mineral_kinetics=db.get_mineral_reactions(["Calcite(s)"], ["default"]),
    )

    params: TstParameters = network.tst_params

    assert params.stoich.loc["Calcite(s)", "Ca++"] == pytest.approx(1.1)
    assert params.stoich.loc["Calcite(s)", "CO3--"] == pytest.approx(0.5)
    assert params.stoich.loc["Calcite(s)", "Calcite(s)"] == 0.0
    assert params.stoich.loc["Calcite(s)", "H+"] == 0.0
    assert params.dep.loc["Calcite(s)"].eq(0.0).all()
    assert params.min_eq_const["Calcite(s)"] == pytest.approx(KSP_CALCITE)
