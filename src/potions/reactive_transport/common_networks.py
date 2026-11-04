from potions.core import (
    ExchangeReaction,
    ReactionNetwork,
    ChemicalDatabase,
    MineralSpecies,
    PrimaryAqueousSpecies,
    SecondarySpecies,
    MineralKineticData,
)


def get_carbon_network() -> ReactionNetwork:
    # The default database with all of the species
    db = ChemicalDatabase.load_default()
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


def get_simple_carbon_network() -> ReactionNetwork:
    # The default database with all of the species
    db = ChemicalDatabase.load_default()
    primary_species_names: list[str] = ["DOC"]
    secondary_species_names: list[str] = []
    mineral_species_names: list[str] = ["SOC(s)"]
    mineral_kinetics_names: list[str] = ["test"]
    exchange_species_names: list[str] = []

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


def get_simple_carbon_adsorption_network() -> ReactionNetwork:
    # The default database with all of the species
    db = ChemicalDatabase.load_default()
    primary_species_names: list[str] = ["DOC"]
    secondary_species_names: list[str] = []
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
    # The default database with all of the species
    db = ChemicalDatabase.load_default()
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


def get_soc_calcite_network() -> ReactionNetwork:
    """
    Build the SOC / DOC / calcite reaction network.

    Assembles a complete :class:`ReactionNetwork` from the default
    potions chemical database containing:

    - primary aqueous species: ``DOC``, ``Ca++``, ``HCO3-``, ``H+``
    - secondary (speciation) species: ``CO2(aq)`` and ``CO3--``
    - minerals with kinetics: ``SOC(s)`` (Monod), ``SOC-lz(s)``
      (Monod), and ``Calcite(s)`` (TST)
    - exchange species: ``XDOC`` (DOC adsorption onto a reactive
      mineral surface)

    ``HCO3-`` and ``H+`` are included because the CO2(aq) and CO3--
    speciation equilibria reference them.  The DOC-consuming reaction
    (``SOC-lz(s)``) emits ``CO2(aq)``, and the ``CO3--`` produced by
    calcite dissolution is speciated through those same equilibria.

    Returns:
        The assembled reaction network.
    """
    # The default database with all of the species
    db = ChemicalDatabase.load_default()
    primary_species_names: list[str] = ["DOC", "Ca++", "HCO3-", "H+"]
    secondary_species_names: list[str] = ["CO2(aq)", "CO3--"]
    mineral_species_names: list[str] = ["SOC(s)", "SOC-lz(s)", "Calcite(s)"]
    mineral_kinetics_names: list[str] = ["test", "test", "default"]
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
    mineral_kinetics: MineralKineticData = db.get_mineral_reactions(
        mineral_species_names, labels=mineral_kinetics_names
    )
    exchange_species: list[ExchangeReaction] = db.get_exchange_reactions(
        exchange_species_names
    )

    network: ReactionNetwork = ReactionNetwork(
        primary_aqueous=primary_species,
        mineral=mineral_species,
        mineral_kinetics=mineral_kinetics,
        secondary=secondary_species,
        exchange_species=exchange_species,
    )
    return network
