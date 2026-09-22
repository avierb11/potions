import numpy as np
import pytest

from ..reactive_transport.database import (
    ChemicalDatabase,
    PrimaryAqueousSpecies,
    SecondarySpecies,
)
from ..reactive_transport.kinetic_structures import (
    EquilibriumParameters,
    MonodParameters,
    PARAMETERS_PER_MINERAL,
    TstParameters,
)
from ..reactive_transport.reaction_network import ReactionNetwork


@pytest.fixture(scope="module")
def db() -> ChemicalDatabase:
    return ChemicalDatabase.load_default()


@pytest.fixture(scope="module")
def primary(db: ChemicalDatabase) -> list[PrimaryAqueousSpecies]:
    return db.get_primary_aqueous_species(["DOC", "HCO3-", "H+"])


@pytest.fixture(scope="module")
def secondary(db: ChemicalDatabase) -> list[SecondarySpecies]:
    return db.get_secondary_species(["CO2(aq)", "CO3--"])


@pytest.fixture(scope="module")
def monod_network(db, primary, secondary) -> ReactionNetwork:
    """A carbon network where the mineral (SOC(s)) uses Monod kinetics."""
    mineral = db.get_mineral_species(["SOC(s)"])
    kin = db.get_mineral_reactions(["SOC(s)"], ["test"])
    return ReactionNetwork(
        primary_aqueous=primary, mineral=mineral, secondary=secondary, mineral_kinetics=kin
    )


@pytest.fixture(scope="module")
def tst_network(db, primary, secondary) -> ReactionNetwork:
    """A network where the mineral (Gypsum) uses TST kinetics."""
    mineral = db.get_mineral_species(["Gypsum"])
    label = next(iter(db.tst_reactions["Gypsum"]))
    kin = db.get_mineral_reactions(["Gypsum"], [label])
    return ReactionNetwork(
        primary_aqueous=primary, mineral=mineral, secondary=secondary, mineral_kinetics=kin
    )


def test_species_order_is_primary_then_secondary_then_mineral(
    monod_network: ReactionNetwork,
) -> None:
    order = monod_network.species_order
    # Primary species come first, then secondary, then the mineral.
    assert order[:3] == ["DOC", "HCO3-", "H+"]
    assert order[3:5] == ["CO2(aq)", "CO3--"]
    assert order[5] == "SOC(s)"


def test_species_names_equals_species_order(monod_network: ReactionNetwork) -> None:
    assert monod_network.species_names == monod_network.species_order


def test_species_types(monod_network: ReactionNetwork) -> None:
    types = dict(zip(monod_network.species.index, monod_network.species["type"]))
    assert types["DOC"] == "primary"
    assert types["H+"] == "primary"
    assert types["CO2(aq)"] == "secondary"
    assert types["SOC(s)"] == "mineral"


def test_equilibrium_species_are_aqueous(monod_network: ReactionNetwork) -> None:
    eq = monod_network.equilibrium_species
    # 3 primary + 2 secondary = 5 aqueous species.
    assert len(eq) == 5
    assert set(eq["type"]).issubset({"primary", "secondary"})
    assert "SOC(s)" not in eq.index


def test_kinetic_species_include_minerals(monod_network: ReactionNetwork) -> None:
    kin = monod_network.kinetic_species
    assert "SOC(s)" in kin.index


def test_charges_match_specification(monod_network: ReactionNetwork) -> None:
    # Charges are a pandas Series indexed by species order.
    charges = monod_network.charges
    assert charges["DOC"] == 0.0
    assert charges["H+"] == 1.0
    assert charges["HCO3-"] == -1.0
    assert charges["CO3--"] == -2.0
    assert charges["SOC(s)"] == 0.0


def test_has_exchange_is_false(monod_network: ReactionNetwork) -> None:
    assert monod_network.has_exchange is False


def test_mineral_species_names(monod_network: ReactionNetwork) -> None:
    assert monod_network.mineral_species_names == ["SOC(s)"]


def test_num_mineral_parameters(monod_network: ReactionNetwork) -> None:
    assert monod_network.num_minerals == 1
    assert monod_network.num_mineral_parameters == PARAMETERS_PER_MINERAL


def test_num_aqueous_species(monod_network: ReactionNetwork) -> None:
    assert monod_network.num_aqueous_species == 5


def test_transport_mask_is_immobile_for_mineral(monod_network: ReactionNetwork) -> None:
    order = monod_network.species_order
    mask = {name: bool(v) for name, v in zip(order, monod_network.transport_mask)}
    assert mask["DOC"] is True  # aqueous species are mobile
    assert mask["SOC(s)"] is False  # minerals are immobile


def test_mineral_molar_masses(monod_network: ReactionNetwork) -> None:
    masses = monod_network.mineral_molar_masses
    assert len(masses) == 1


def test_rate_consts_is_positive(monod_network: ReactionNetwork) -> None:
    rates = monod_network.rate_consts
    assert len(rates) == 1
    assert rates[0] > 0


def test_mineral_stoichiometry_columns(monod_network: ReactionNetwork) -> None:
    stoich = monod_network.mineral_stoichiometry
    assert "SOC(s)" in stoich.columns


def test_equilibrium_parameters_constructed(monod_network: ReactionNetwork) -> None:
    eq = monod_network.equilibrium_parameters
    assert isinstance(eq, EquilibriumParameters)
    # The stoich matrix covers all species in order.
    assert eq.stoich.shape[1] == len(monod_network.species_order)


def test_monod_params_constructed(monod_network: ReactionNetwork) -> None:
    params = monod_network.monod_params
    assert isinstance(params, MonodParameters)
    # Rows = number of minerals, columns = all species.
    assert params.monod_mat.shape[0] == 1
    assert params.monod_mat.shape[1] == len(monod_network.species_order)


def test_tst_params_constructed(tst_network: ReactionNetwork) -> None:
    params = tst_network.tst_params
    assert isinstance(params, TstParameters)
    assert params.stoich.shape[0] == 1  # one mineral
    assert params.stoich.shape[1] == len(tst_network.species_order)


def test_tst_rate_consts_match_logk(tst_network: ReactionNetwork, db) -> None:
    # For a TST mineral, rate_const = 10^(rate_constant), where rate_constant is
    # the log10 rate stored on the TST reaction.
    label = next(iter(db.tst_reactions["Gypsum"]))
    expected = 10.0 ** (db.tst_reactions["Gypsum"][label].rate_constant)
    assert tst_network.rate_consts[0] == pytest.approx(expected)


def test_get_default_aqueous_initial_state(
    monod_network: ReactionNetwork,
) -> None:
    state = monod_network.get_default_aqueous_initial_state()
    assert state.shape == (5,)
    assert np.all(state == 1e-6)

    custom = monod_network.get_default_aqueous_initial_state(init_conc=5e-7)
    assert np.all(custom == 5e-7)
