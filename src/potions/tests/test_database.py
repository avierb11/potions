import pytest

from ..reactive_transport.database import (
    ChemicalDatabase,
    MineralKineticData,
    MineralKineticReaction,
    TstReaction,
    MonodReaction,
)


@pytest.fixture(scope="module")
def db() -> ChemicalDatabase:
    return ChemicalDatabase.load_default()


def test_load_default_has_species(db: ChemicalDatabase) -> None:
    assert len(db.primary_species) > 0
    assert len(db.secondary_species) > 0
    assert len(db.mineral_species) > 0


def test_get_primary_aqueous_species_single(db: ChemicalDatabase) -> None:
    # The getter takes an iterable; a single element returns a one-item list.
    out = db.get_primary_aqueous_species(["DOC"])
    assert len(out) == 1
    assert out[0].name == "DOC"


def test_get_primary_aqueous_species_iterable(db: ChemicalDatabase) -> None:
    out = db.get_primary_aqueous_species(["DOC", "H+", "HCO3-"])
    assert [x.name for x in out] == ["DOC", "H+", "HCO3-"]


def test_get_primary_aqueous_species_missing_raises(db: ChemicalDatabase) -> None:
    with pytest.raises(KeyError):
        db.get_primary_aqueous_species("__definitely_not_a_species__")


def test_get_secondary_species(db: ChemicalDatabase) -> None:
    # The getter takes an iterable (it does not special-case a bare string).
    out = db.get_secondary_species(["CO3--"])
    assert len(out) == 1
    assert out[0].name == "CO3--"


def test_get_mineral_species_single_and_iterable(db: ChemicalDatabase) -> None:
    name = next(iter(db.mineral_species))
    assert db.get_mineral_species(name)[0].name == name
    assert [x.name for x in db.get_mineral_species([name, name])] == [name, name]


def test_get_mineral_species_missing_raises(db: ChemicalDatabase) -> None:
    with pytest.raises(KeyError):
        db.get_mineral_species("__not_a_mineral__")


def test_get_single_mineral_reaction_tst(db: ChemicalDatabase) -> None:
    mineral = next(iter(db.tst_reactions))
    label = next(iter(db.tst_reactions[mineral]))
    reaction_type, reaction = db.get_single_mineral_reaction(mineral, label)
    assert reaction_type == "tst"
    assert isinstance(reaction, TstReaction)
    assert reaction.mineral_name == mineral


def test_get_single_mineral_reaction_monod(db: ChemicalDatabase) -> None:
    mineral = next(iter(db.monod_reactions))
    label = next(iter(db.monod_reactions[mineral]))
    reaction_type, reaction = db.get_single_mineral_reaction(mineral, label)
    assert reaction_type == "monod"
    assert isinstance(reaction, MonodReaction)


def test_get_single_mineral_reaction_accepts_species_object(
    db: ChemicalDatabase,
) -> None:
    # Pick a mineral that has a TST reaction and a matching mineral_species entry,
    # so it can be passed back as a MineralSpecies object.
    common = next(m for m in db.tst_reactions if m in db.mineral_species)
    species = db.mineral_species[common]
    label = next(iter(db.tst_reactions[common]))
    reaction_type, _ = db.get_single_mineral_reaction(species, label)
    assert reaction_type == "tst"


def test_get_single_mineral_reaction_missing_raises(db: ChemicalDatabase) -> None:
    with pytest.raises(ValueError):
        db.get_single_mineral_reaction("__not_a_mineral__", "default")


def test_get_mineral_reactions_returns_kinetic_data(db: ChemicalDatabase) -> None:
    monod_mineral = next(iter(db.monod_reactions))
    label = next(iter(db.monod_reactions[monod_mineral]))
    kin = db.get_mineral_reactions([monod_mineral], [label])
    assert isinstance(kin, MineralKineticData)
    assert monod_mineral in kin.monod_reactions
    assert kin.tst_reactions == {}


def test_round_trip_to_file_from_file(
    db: ChemicalDatabase, tmp_path
) -> None:
    path = tmp_path / "db_roundtrip.json"
    db.to_file(str(path))
    assert path.exists()

    restored = ChemicalDatabase.from_file(str(path))
    assert restored.primary_species == db.primary_species
    assert restored.secondary_species == db.secondary_species
    assert restored.mineral_species == db.mineral_species
    assert restored.tst_reactions == db.tst_reactions
    assert restored.monod_reactions == db.monod_reactions


def test_mineral_kinetic_reaction_is_frozen() -> None:
    r = MineralKineticReaction(mineral_name="m", label="l", rate_constant=1.0)
    with pytest.raises(Exception):
        r.rate_constant = 2.0  # type: ignore[misc]


def _is_nonempty_str(s: str) -> bool:
    return isinstance(s, str) and len(s) > 0


def test_database_field_types(db: ChemicalDatabase) -> None:
    for mapping in (db.primary_species, db.mineral_species):
        assert all(_is_nonempty_str(k) for k in mapping)


def test_tst_reaction_is_subclass_of_kinetic_reaction() -> None:
    assert issubclass(TstReaction, MineralKineticReaction)
