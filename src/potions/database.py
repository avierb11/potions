"""
This file contains the types representing database objects
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json

from .utils import Matrix


@dataclass(frozen=True)
class PrimaryAqueousSpecies:
    name: str  # Name of the species, like "DOC", "Ca++", ...
    molar_mass: float  # Molar mass, in grams per mole, of this species
    charge: float  # Charge of this species
    dh_size_param: float  # Debye-Huckel size parameter


@dataclass(frozen=True)
class SecondarySpecies:
    name: str  # Name of the species
    stoichiometry: dict[
        str, float
    ]  # Stoichiometry of this equation in terms of primary species
    eq_consts: list[float]  # Equilibrium constants for this equation
    dh_size_param: float  # Debye-Huckel size parameter
    charge: float  # Charge of this species
    molar_mass: float  # Molar mass, in grams per mole, of this species


@dataclass(frozen=True)
class MineralSpecies:
    name: str  # Name of the species
    molar_mass: float  # Molar mass, in grams per mole, of this species
    stoichiometry: dict[str, float]  # Stiochiometry describing this mineral species
    eq_consts: list[float]
    molar_volume: float  # Molar volume of this mineral, in grams per mole


PrimarySpecies = PrimaryAqueousSpecies | MineralSpecies


@dataclass(frozen=True)
class SurfaceComplexationReaction:
    pass


@dataclass(frozen=True)
class MineralKineticReaction:
    mineral_name: str
    label: str  # Label of this reaction in the database, in case there are others
    rate_constant: float  # Rate constant at 25 C (298 K)


@dataclass(frozen=True)
class TstReaction(MineralKineticReaction):
    dependence: dict[str, float]  # Dependence of these reactions on other species


@dataclass(frozen=True)
class MonodReaction(MineralKineticReaction):
    monod_terms: dict[str, float]
    inhib_terms: dict[str, float]


@dataclass(frozen=True)
class ExchangeReaction:
    name: str  # Name of the species, like "XDOC", ...
    stoichiometry: dict[str, float]  # Stoichiometry describing this reaction
    dh_size_param: float  # Debye-Huckel size parameter
    charge: float  # Charge of this species


# ==== Class containing the entire set of species ==== #
@dataclass
class ChemicalDatabase:
    """
    A container class to hold all chemical species and reaction definitions.

    Attributes:
        primary_species: A list of `PrimarySpecies` objects.
        secondary_species: A list of `SecondarySpecies` objects.
        mineral_species: A list of `MineralSpecies` objects.
        exchange_reactions: A list of `ExchangeReaction` objects.
        tst_reactions: A list of `TstReaction` objects.
        monod_reactions: A list of `MonodReaction` objects.
        surface_complexation_reactions: A list of `SurfaceComplexationReaction` objects.
    """

    primary_species: dict[str, PrimaryAqueousSpecies]
    secondary_species: dict[str, SecondarySpecies]
    mineral_species: dict[str, MineralSpecies]
    exchange_reactions: dict[str, ExchangeReaction]
    tst_reactions: dict[
        str, dict[str, TstReaction]
    ]  # First key is Mineral name, second key is reaction label
    monod_reactions: dict[str, dict[str, MonodReaction]]

    @staticmethod
    def load_database() -> ChemicalDatabase:
        """Load the default database with all of the reactions it contains"""
        return NotImplemented

    def to_file(self, file_path: str) -> None:
        """
        Save this database object to a file
        """
        with open(file_path, "w+") as f:
            json.dump(self, f, indent=2)

    @staticmethod
    def from_file(file_path: str) -> ChemicalDatabase:
        """
        Load a ChemicalDatabase object from a file
        """
        with open(file_path, "r") as f:
            raw_dict: dict = json.load(f)

        primary_dict: dict = raw_dict["primary_species"]
        secondary_dict: dict = raw_dict["secondary_species"]
        mineral_dict: dict = raw_dict["mineral_species"]
        tst_dict: dict = raw_dict["tst_dict"]
        monod_dict: dict = raw_dict["monod_dict"]
        exchange_dict: dict = raw_dict["exchange_reactions"]

        # Construct Python objects from Database
        primary_species: dict[str, PrimaryAqueousSpecies] = {
            key: PrimaryAqueousSpecies(**vals) for key, vals in primary_dict.items()
        }
        secondary_species: dict[str, SecondarySpecies] = {
            key: SecondarySpecies(**vals) for key, vals in secondary_dict.items()
        }
        mineral_species: dict[str, MineralSpecies] = {
            key: MineralSpecies(**vals) for key, vals in mineral_dict.items()
        }
        exchange_reactions: dict[str, ExchangeReaction] = {
            key: ExchangeReaction(**vals) for key, vals in exchange_dict.items()
        }

        tst_reactions: dict[str, dict[str, TstReaction]] = {}
        for key, val in tst_dict.items():
            tst_reactions[key] = {
                label: TstReaction(**params) for label, params in val.items()
            }

        monod_reactions: dict[str, dict[str, MonodReaction]] = {}
        for key, val in monod_dict.items():
            monod_reactions[key] = {
                label: MonodReaction(**params) for label, params in val.items()
            }

        return ChemicalDatabase(
            primary_species=primary_species,
            secondary_species=secondary_species,
            mineral_species=mineral_species,
            tst_reactions=tst_reactions,
            monod_reactions=monod_reactions,
            exchange_reactions=exchange_reactions
        )

    def get_primary_species(
        self, species_name: str | list[str]
    ) -> list[PrimarySpecies]:
        """Get one or more primary species from the database, including aqueous or mineral species"""
        return NotImplemented

    def get_primary_aqueous_species(
        self, species_name: str | list[str]
    ) -> list[PrimaryAqueousSpecies]:
        """Get one or more mineral species from the database"""
        return [self.primary_species[name] for name in species_name]

    def get_mineral_species(
        self, primary_aq: list[PrimaryAqueousSpecies], species_name: str | list[str]
    ) -> list[MineralSpecies]:
        """Get one or more mineral species from the database"""
        return [self.mineral_species[name] for name in species_name]

    def get_secondary_species(
        self, primary: list[PrimarySpecies], species_name: str | list[str]
    ) -> list[SecondarySpecies]:
        """Get the secondary species by also providing the primary species to ensure that the reactions are valid.
        If a secondary species contains a primary species not included in `primary`, an error will be thrown to prevent
        the user from defining impossible reaction networks
        """
        return [self.secondary_species[name] for name in species_name]

    def get_single_mineral_reaction(
        self, primary: list[PrimarySpecies], mineral: str | MineralSpecies, label: str
    ) -> MineralKineticReaction:
        """
        Select the mineral reaction parameters produced by the included metrics
        """
        return NotImplemented

    def get_mineral_reactions(
        self,
        primary: list[PrimarySpecies],
        mineral: list[str] | list[MineralSpecies],
        labels: list[str],
    ) -> list[MineralKineticReaction]:
        """
        Select the mineral reaction parameters for multiple reactions
        """
        return NotImplemented


@dataclass
class ReactionNetwork:
    primary_aqueous: list[PrimaryAqueousSpecies]
    mineral: list[MineralSpecies]
    secondary: list[SecondarySpecies]
    mineral_kinetics: list[MineralKineticReaction]

    def species_names(self) -> list[str]:
        """
        Return the names of the species, in order, that they are solved
        """
        return NotImplemented

    def stoich_matrix(self) -> Matrix:
        """
        Return the matrix describing the stoichiometry of each reaction in the network.
        This has the dimension (number of minerals x number of primary species)
        """
        raise NotImplementedError()

    def mass_conservation_matrix(self) -> Matrix:
        """
        Return the mass conservation matrix, describing how mass and charge are balanced
        between reactions.
        This has the dimension (number of primary species x number of total species)
        """
        return NotImplemented

    def equilibrium_matrix(self) -> Matrix:
        """
        Construct and return the matrix describing the equilibrium constants of the secondary
        species
        This has the shape (number of secondary species x number of chemical species)
        """
        return NotImplemented
