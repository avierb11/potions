"""
This file contains the types representing database objects
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import os
from typing import Literal

from pandas import DataFrame, Series

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


@dataclass
class MineralKineticData:
    tst_reactions: dict[str, TstReaction]
    monod_reactions: dict[str, MonodReaction]


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
    log10_k_eq: float  # Base-10 log of the equilibrium constant
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
    def load_default() -> ChemicalDatabase:
        """
        """
        db_file_path: str = os.path.join(
            os.path.dirname(__file__), "default_database.json"
        )

        return ChemicalDatabase.from_file(db_file_path)


    def to_file(self, file_path: str) -> None:
        """
        Save this database object to a file
        """
        with open(file_path, "w+") as f:
            json.dump(asdict(self), f, indent=2)


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
        tst_dict: dict = raw_dict["tst_reactions"]
        monod_dict: dict = raw_dict["monod_reactions"]
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


    def get_primary_aqueous_species(
        self, species_name: str | list[str]
    ) -> list[PrimaryAqueousSpecies]:
        """Get one or more mineral species from the database"""
        return [self.primary_species[name] for name in species_name]


    def get_mineral_species(
        self, mineral_name: str | list[str]
    ) -> list[MineralSpecies]:
        """Get one or more mineral species from the database"""
        return [self.mineral_species[name] for name in mineral_name]


    def get_secondary_species(
        self, species_name: str | list[str]
    ) -> list[SecondarySpecies]:
        """Get the secondary species by also providing the primary species to ensure that the reactions are valid.
        If a secondary species contains a primary species not included in `primary`, an error will be thrown to prevent
        the user from defining impossible reaction networks
        """
        return [self.secondary_species[name] for name in species_name]


    def get_single_mineral_reaction(
        self, mineral: str | MineralSpecies, label: str
    ) -> tuple[Literal["tst", "monod"], MineralKineticReaction]:
        """
        Select the mineral reaction parameters produced by the included metrics
        """
        mineral_name: str
        if isinstance(mineral, MineralSpecies):
            mineral_name = mineral.name
        elif isinstance(mineral, str):
            mineral_name = mineral
        else:
            raise TypeError(f"Mineral {mineral} is not a string or MineralSpecies")

        if mineral_name in self.tst_reactions:
            return "tst", self.tst_reactions[mineral_name][label]
        elif mineral_name in self.monod_reactions:
            return "monod", self.monod_reactions[mineral_name][label]
        else:
            raise ValueError(f"Mineral {mineral_name} not found in either TST or Monod reactions")


    def get_mineral_reactions(
        self,
        mineral: list[str] | list[MineralSpecies],
        labels: list[str],
    ) -> MineralKineticData:
        """
        Select the mineral reaction parameters for multiple reactions
        """
        mineral_names: list[str] = [mineral.name if isinstance(mineral, MineralSpecies) else mineral for mineral in mineral]

        mineral_reactions: dict = {
            "tst": [],
            "monod": []
        }

        for mineral, label in zip(mineral_names, labels): # type: ignore
            reaction_type, reaction = self.get_single_mineral_reaction(mineral, label) # type: ignore
            mineral_reactions[reaction_type].append(reaction)

        return MineralKineticData(
            tst_reactions={x.mineral_name: x for x in mineral_reactions["tst"]},
            monod_reactions={x.mineral_name: x for x in mineral_reactions["monod"]}
        )
    
    def get_exchange_reactions(
        self,
        species_name: str | list[str],
    ) -> list[ExchangeReaction]:
        """
        Select the mineral reaction parameters for multiple reactions
        """
        if isinstance(species_name, str):
            species_name = [species_name]

        return [self.exchange_reactions[name] for name in species_name]
       


