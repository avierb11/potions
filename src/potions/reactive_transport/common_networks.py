from .reaction_network import ReactionNetwork
from .database import (
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

    primary_species: list[PrimaryAqueousSpecies] = db.get_primary_aqueous_species(
        primary_species_names
    )
    secondary_species: list[SecondarySpecies] = db.get_secondary_species(
        secondary_species_names
    )
    mineral_species: list[MineralSpecies] = db.get_mineral_species(
        mineral_species_names
    )
    min_kin = db.get_mineral_reactions(mineral_species, labels=mineral_kinetics_names)
    network: ReactionNetwork = ReactionNetwork(
        primary_aqueous=primary_species,
        mineral=mineral_species,
        mineral_kinetics=min_kin,
        secondary=secondary_species,
    )
    return network
