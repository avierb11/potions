"""Tests for the informative ``__repr__`` implementations and the
species-viewing utilities on the compiled (``potions.core``) reactive
transport types.

The pure-Python dataclass twins in ``potions.reactive_transport`` are
covered by ``test_database.py`` / ``test_kinetic_structures.py``; this
module exercises the Rust implementations directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..core import (
    ChemicalDatabase,
    MineralAuxParams,
    MineralParameters,
    MineralSpecies,
    MonodParameters,
    MonodReaction,
    ReactionNetwork,
    RiverDimensions,
    RiverParameters,
    RiverZone,
    RtParameters,
    RtStep,
    RtZone,
    TstParameters,
    TstReaction,
    ZoneDimensions,
)
from ..reactive_transport.common_networks import (
    get_carbon_network,
    get_chloride_network,
)


# ===================== Fixtures ===================== #
@pytest.fixture(scope="module")
def db() -> ChemicalDatabase:
    return ChemicalDatabase.load_default()


@pytest.fixture(scope="module")
def carbon_network() -> ReactionNetwork:
    return get_carbon_network()


@pytest.fixture(scope="module")
def chloride_network() -> ReactionNetwork:
    return get_chloride_network()


@pytest.fixture(scope="module")
def rt_zone(carbon_network: ReactionNetwork) -> RtZone:
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    dim = ZoneDimensions(20.0, 10.0)
    return RtZone(
        carbon_network,
        RtParameters(dim, mp),
        do_reactions=True,
        do_speciation=True,
        name="testzone",
    )


@pytest.fixture(scope="module")
def river_zone(carbon_network: ReactionNetwork) -> RiverZone:
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    dim = RiverDimensions(2.0, 5.0)
    return RiverZone(
        carbon_network,
        RiverParameters(dim, mp),
        do_reactions=False,
        do_speciation=True,
        name="riverzone",
    )


# ===================== ChemicalDatabase ===================== #
def test_database_repr_reports_counts(db: ChemicalDatabase) -> None:
    assert repr(db) == (
        "ChemicalDatabase(primary_species=100,secondary_species=1699,"
        "mineral_species=1153,exchange_reactions=33,tst_reactions=68,"
        "monod_reactions=9)"
    )


def test_database_summary_sections(db: ChemicalDatabase) -> None:
    lines = db.summary().split("\n")
    assert lines[0] == (
        "ChemicalDatabase summary: 100 primary aqueous, 1699 secondary, "
        "1153 mineral, 33 exchange, 68 TST minerals, 9 Monod minerals"
    )
    assert len(lines) == 7

    sections: list[tuple[str, list[str]]] = [
        ("primary aqueous", db.get_primary_aqueous_species_names()),
        ("secondary", db.get_secondary_species_names()),
        ("mineral", db.get_mineral_species_names()),
        ("exchange", db.get_exchange_species_names()),
        ("TST minerals", db.get_tst_mineral_names()),
        ("Monod minerals", db.get_monod_mineral_names()),
    ]
    for (label, names), line in zip(sections, lines[1:]):
        assert label in line
        # Preview of the first five sorted names, truncated with "...".
        assert ", ".join(names[:5]) in line
        assert line.endswith("...)")


def test_database_name_getters_match_fields(db: ChemicalDatabase) -> None:
    # Each getter is the (sorted) key set of the corresponding field.
    assert db.get_primary_aqueous_species_names() == sorted(db.primary_species)
    assert db.get_secondary_species_names() == sorted(db.secondary_species)
    assert db.get_mineral_species_names() == sorted(db.mineral_species)
    assert db.get_exchange_species_names() == sorted(db.exchange_reactions)
    assert db.get_tst_mineral_names() == sorted(db.tst_reactions)
    assert db.get_monod_mineral_names() == sorted(db.monod_reactions)


def test_database_name_getters_content(db: ChemicalDatabase) -> None:
    assert "DOC" in db.get_primary_aqueous_species_names()
    assert "CO2(aq)" in db.get_secondary_species_names()
    assert "SOC(s)" in db.get_mineral_species_names()
    assert "XDOC" in db.get_exchange_species_names()
    assert "Gypsum" in db.get_tst_mineral_names()
    assert "SOC(s)" in db.get_monod_mineral_names()


# ===================== Database species ===================== #
def test_primary_aqueous_species_repr(db: ChemicalDatabase) -> None:
    doc = db.get_primary_aqueous_species(["DOC"])[0]
    assert repr(doc) == (
        "PrimaryAqueousSpecies(name=DOC,molar_mass=300,charge=0,dh_size_param=4)"
    )


def test_secondary_species_repr(db: ChemicalDatabase) -> None:
    co2 = db.get_secondary_species(["CO2(aq)"])[0]
    assert repr(co2) == (
        "SecondarySpecies(name=CO2(aq),"
        "stoichiometry={'CO2(aq)': -1, 'H+': 1, 'H2O': -1, 'HCO3-': 1},"
        "eq_consts=[-6.5804, -6.3447, -6.2684, -6.3882, -6.7235, -7.1969, "
        "-7.7868, -8.528],"
        "dh_size_param=3,charge=0,molar_mass=44.0098)"
    )


def test_mineral_species_repr(db: ChemicalDatabase) -> None:
    soc = db.get_mineral_species(["SOC(s)"])[0]
    assert repr(soc) == (
        "MineralSpecies(name=SOC(s),molar_mass=180,"
        "stoichiometry={'CO2(aq)': 0.6, 'DOC': 0.55, 'SOC(s)': -1},"
        "eq_consts=[-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],"
        "molar_volume=115.48)"
    )


def test_exchange_reaction_repr(db: ChemicalDatabase) -> None:
    xdoc = db.get_exchange_reactions(["XDOC"])[0]
    assert repr(xdoc) == (
        "ExchangeReaction(name=XDOC,stoichiometry={'DOC': 1, 'X-': 1, "
        "'XDOC': -1},log10_k_eq=-2.75,charge=0)"
    )


def test_tst_reaction_repr_from_database(db: ChemicalDatabase) -> None:
    gypsum = db.tst_reactions["Gypsum"]["default"]
    assert repr(gypsum) == (
        "TstReaction(mineral_name=Gypsum,label=default,rate_constant=-6,"
        "dependence={})"
    )


def test_tst_reaction_repr_constructed() -> None:
    rxn = TstReaction("Calcite(s)", "default", -3.5, {"H+": 1.0, "Ca++": 0.5})
    assert repr(rxn) == (
        "TstReaction(mineral_name=Calcite(s),label=default,rate_constant=-3.5,"
        "dependence={'Ca++': 0.5, 'H+': 1})"
    )


def test_monod_reaction_repr_from_database(db: ChemicalDatabase) -> None:
    soc = db.monod_reactions["SOC(s)"]["test"]
    assert repr(soc) == (
        "MonodReaction(mineral_name=SOC(s),label=test,rate_constant=-10,"
        "monod_terms={'SOC(s)': 0.000006},inhib_terms={})"
    )


def test_monod_reaction_repr_constructed() -> None:
    rxn = MonodReaction(
        "SOC(s)", "test", -4.0, {"DOC": 2.0}, {"Ca++": 0.25}
    )
    assert repr(rxn) == (
        "MonodReaction(mineral_name=SOC(s),label=test,rate_constant=-4,"
        "monod_terms={'DOC': 2},inhib_terms={'Ca++': 0.25})"
    )


def test_mineral_kinetic_data_repr(db: ChemicalDatabase) -> None:
    monod_only = db.get_mineral_reactions(["SOC(s)"], ["test"])
    assert repr(monod_only) == (
        "MineralKineticData(tst_reactions=[],monod_reactions=['SOC(s)'])"
    )
    tst_only = db.get_mineral_reactions(["Gypsum"], ["default"])
    assert repr(tst_only) == (
        "MineralKineticData(tst_reactions=['Gypsum'],monod_reactions=[])"
    )


# ===================== Kinetic structures ===================== #
def test_mineral_aux_params_repr() -> None:
    aux = MineralAuxParams(1.0, 2.0, 1.5, 10.0)
    assert repr(aux) == "MineralAuxParams(sw_threshold=1,sw_exp=2,q_10=1.5,reactivity=10)"


def test_zone_dimensions_repr() -> None:
    dim = ZoneDimensions(20.0, 10.0)
    assert repr(dim) == "ZoneDimensions(max_water=20,passive_water_storage=10)"


def test_river_dimensions_repr() -> None:
    dim = RiverDimensions(2.0, 5.0)
    assert repr(dim) == "RiverDimensions(bed_depth=2,passive_water_storage=5)"


def test_mineral_parameters_repr() -> None:
    one = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    assert repr(one) == "MineralParameters(num_minerals=1)"

    three = MineralParameters(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    assert repr(three) == "MineralParameters(num_minerals=3)"


def test_rt_parameters_repr() -> None:
    dim = ZoneDimensions(20.0, 10.0)
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    assert repr(RtParameters(dim, None)) == (
        "RtParameters(dimensions=ZoneDimensions(max_water=20,"
        "passive_water_storage=10),mineral_params=None)"
    )
    assert repr(RtParameters(dim, mp)) == (
        "RtParameters(dimensions=ZoneDimensions(max_water=20,"
        "passive_water_storage=10),mineral_params=MineralParameters(num_minerals=1))"
    )


def test_river_parameters_repr() -> None:
    dim = RiverDimensions(2.0, 5.0)
    mp = MineralParameters(
        np.array([0.1]), np.array([2.0]), np.array([1.5]), np.array([10.0])
    )
    assert repr(RiverParameters(dim, None)) == (
        "RiverParameters(dimensions=RiverDimensions(bed_depth=2,"
        "passive_water_storage=5),mineral_params=None)"
    )
    assert repr(RiverParameters(dim, mp)) == (
        "RiverParameters(dimensions=RiverDimensions(bed_depth=2,"
        "passive_water_storage=5),mineral_params=MineralParameters(num_minerals=1))"
    )


def test_monod_parameters_repr(carbon_network: ReactionNetwork) -> None:
    # One Monod mineral (SOC(s)) across the carbon network's eight species.
    monod: MonodParameters = carbon_network.monod_params
    assert repr(monod) == "MonodParameters(num_minerals=1,num_species=8)"


def test_tst_parameters_repr(carbon_network: ReactionNetwork) -> None:
    # Every network mineral gets a (zero-filled) TST column.
    tst: TstParameters = carbon_network.tst_params
    assert repr(tst) == "TstParameters(num_minerals=1,num_species=8)"


def test_equilibrium_parameters_repr(carbon_network: ReactionNetwork) -> None:
    eq = carbon_network.equilibrium_parameters
    assert repr(eq) == "EquilibriumParameters(num_species=8,num_total=5,num_free=5)"

    # Cross-check the reported dimensions against the underlying matrices.
    total = np.asarray(eq.get_total_mat())
    null_space = np.asarray(eq.get_stoich_null_space())
    assert repr(eq) == (
        f"EquilibriumParameters(num_species={total.shape[1]},"
        f"num_total={total.shape[0]},num_free={null_space.shape[1]})"
    )


# ===================== Reaction networks ===================== #
def test_reaction_network_repr_carbon(carbon_network: ReactionNetwork) -> None:
    assert repr(carbon_network) == (
        "ReactionNetwork(num_species=8,num_aqueous_species=5,num_minerals=1,"
        "has_exchange=True,"
        "species=['DOC', 'HCO3-', 'H+', 'CO2(aq)', 'CO3--', 'SOC(s)', 'X-', 'XDOC'])"
    )


def test_reaction_network_repr_no_exchange(chloride_network: ReactionNetwork) -> None:
    assert repr(chloride_network) == (
        "ReactionNetwork(num_species=1,num_aqueous_species=1,num_minerals=0,"
        "has_exchange=False,species=['Cl-'])"
    )


# ===================== Zones ===================== #
def test_rt_zone_repr(rt_zone: RtZone) -> None:
    assert (
        repr(rt_zone)
        == "RtZone(name='testzone',do_reactions=True,do_speciation=True,num_species=8)"
    )


def test_river_zone_repr(river_zone: RiverZone) -> None:
    assert (
        repr(river_zone)
        == "RiverZone(name='riverzone',do_reactions=False,do_speciation=True,num_species=8)"
    )


def test_misc_data_repr(rt_zone: RtZone, river_zone: RiverZone) -> None:
    # Eight species x one mineral; not directly constructible from Python.
    assert repr(rt_zone.misc) == "MiscData(mineral_stoichiometry=8x1,num_species=8)"
    assert repr(river_zone.misc) == "MiscData(mineral_stoichiometry=8x1,num_species=8)"


# ===================== RtStep ===================== #
def test_rt_step_repr() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    step = RtStep(a, b, a, b, a, b, a, b, np.array([0.5]))
    assert repr(step) == (
        "RtStep(state=[1, 2, 3],total_moles=[4, 5, 6],conc_in=[1, 2, 3],"
        "mass_in=[4, 5, 6],lat_conc=[1, 2, 3],vert_conc=[4, 5, 6],"
        "lat_mass=[1, 2, 3],vert_mass=[4, 5, 6],mineral_rates=[0.5])"
    )
