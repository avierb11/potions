from potions.core import (
    ExchangeReaction,
    ReactionNetwork,
    ChemicalDatabase,
    MineralSpecies,
    PrimaryAqueousSpecies,
    SecondarySpecies,
)


def get_carbon_network() -> ReactionNetwork:
    db = ChemicalDatabase.load_default()  # The default database with all of the species
    primary_species_names: list[str] = ["DOC", "HCO3-", "H+"]
    secondary_species_names: list[str] = ["CO2(aq)", "CO3--"]
    mineral_species_names: list[str] = ["SOC(s)"]
    mineral_kinetics_names: list[str] = ["test"]
    exchange_species_names: list[str] = ["XDOC"]

    primary_species: list[PrimaryAqueousSpecies] = db.get_primary_aqueous_species(
        primary_species_names
    )
    secondary_species: list[SecondarySpecies] = db.get_secondary_species(
        secondary_species_names
    )
    mineral_species: list[MineralSpecies] = db.get_mineral_species(
        mineral_species_names
    )
    min_kin = db.get_mineral_reactions(
        mineral_species_names, labels=mineral_kinetics_names
    )
    exchange_species: list[ExchangeReaction] = db.get_exchange_reactions(
        exchange_species_names
    )

    network: ReactionNetwork = ReactionNetwork(
        primary_aqueous=primary_species,
        mineral=mineral_species,
        mineral_kinetics=min_kin,
        secondary=secondary_species,
        exchange_species=exchange_species,
    )
    return network


def get_chloride_network() -> ReactionNetwork:
    db = ChemicalDatabase.load_default()  # The default database with all of the species
    primary_species_names: list[str] = ["Cl-"]
    primary_species: list[PrimaryAqueousSpecies] = db.get_primary_aqueous_species(
        primary_species_names
    )
    min_kin = db.get_mineral_reactions([], labels=[])
    network: ReactionNetwork = ReactionNetwork(
        primary_aqueous=primary_species,
        mineral=[],
        mineral_kinetics=min_kin,
        secondary=[],
        exchange_species=[],
    )
    return network
