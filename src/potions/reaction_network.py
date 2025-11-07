from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from scipy.optimize import fsolve
import scipy.linalg as la
from numpy.typing import NDArray
from .database import ExchangeReaction, MineralKineticData, MineralSpecies, PrimaryAqueousSpecies, SecondarySpecies
from .common_types import Vector, Matrix, ChemicalState, RtForcing

@dataclass(frozen=True)
class MonodParameters:
    """
    The Monod parameters, containing two matrices of shape (number of minerals x number of total species in the simulation)
    """
    monod_mat: DataFrame
    inhib_mat: DataFrame

    def rate(self, all_species_conc: NDArray) -> NDArray:
        """
        Calculate the rate of reaction using Monod kinetics
        """
        monod: NDArray = (all_species_conc / (self.monod_mat + all_species_conc)).sum(axis=1)
        inhib: NDArray = ((self.inhib_mat + all_species_conc) / all_species_conc).sum(axis=1)

        return monod * inhib
    

@dataclass(frozen=True)
class TstParameters:
    """
    The Transition-State Theory parameters, containing two matrices of shape (number of minerals x number of total species in the simulation)
    """
    stoich: DataFrame
    dep: DataFrame
    min_eq_const: Series # Vector of equilibrium constants

    def rate(self, all_species_conc: NDArray) -> NDArray:
        """
        Calculate the rate of reaction using Monod kinetics
        """
        log_prim: NDArray = np.log10(all_species_conc)

        log_dep: NDArray = self.dep.values @ log_prim
        log_iap: NDArray = self.stoich.values @ log_prim

        dep: NDArray = 10 ** log_dep
        iap: NDArray = 10 ** log_iap

        return dep * (1.0 - iap / self.min_eq_const)


@dataclass(frozen=True)
class EquilibriumParameters:
    stoich: DataFrame # Matrix describing the stoichiometry of the secondary species
    equilibrium: Series # Vector of the equilibrium constants for the secondary species
    total: DataFrame # Matrix describing the mass and charge balance of the species 

    @staticmethod
    def from_species(species: Series, primary: list[PrimaryAqueousSpecies], secondary: list[SecondarySpecies], exchange: list[ExchangeReaction]) -> EquilibriumParameters:
        """
        Construct the equilibrium parameters from the database
        """
        raise NotImplementedError()

    @property
    def stoich_null_space(self) -> Matrix:
        """Return the null space of the stoichiometry matrix
        """
        return la.null_space(self.stoich) # type: ignore
    
    @property
    def log10_k_w(self) -> Vector:
        """Return a vector of the equilibrium constants in base-10 logarithm
        """
        return self.equilibrium.map(np.log10).to_numpy()


    @property
    def x_particular(self) -> Vector:
        """
        Return a vector of the particular solution of the null space of the stoichiometry
        """
        return la.pinv(self.stoich) @ self.log10_k_w # type: ignore

    def solve_equilibrium(self, chms: ChemicalState) -> ChemicalState:
        """
        Solve for the equilibrium concentrations of all of the species
        """
        c_tot: Final[Vector[NumTot]] = self.total @ np.concatenate([chms.prim_aq_conc, chms.sec_conc]) # type: ignore


        def conc(x: Vector) -> Vector:
            """
            Return a vector of the aqueous concentrations in base-10 logarithm
            """
            return 10 ** (self.stoich_null_space @ x + self.x_particular) # type: ignore

        def f(x: Vector) -> Vector:
            return c_tot - self.total @ conc(x) # type: ignore

        sol = fsolve(f, x0=np.zeros_like(c_tot))

        new_conc: Vector[NumAqueous] = conc(sol) # type: ignore
        
        return ChemicalState(
            prim_aq_conc=new_conc[:len(chms.prim_aq_conc)], # type: ignore
            min_conc=chms.min_conc,
            sec_conc=new_conc[len(chms.prim_aq_conc):], # type: ignore
        )


class MineralParameters:
    def __init__(self, volume_fraction: NDArray, ssa: NDArray, rate_const: NDArray) -> None:
        self.volume_fraction: NDArray = volume_fraction # Volume fraction of each mineral species in grams mineral/grams solid
        self.ssa: NDArray = ssa # Specific surface area in units of m^2 mineral surface area / grams mineral
        self.rate_const: NDArray = rate_const # Rate constant for the reactions


    @property
    def surface_area(self) -> NDArray:
        return self.volume_fraction * self.ssa # type: ignore


@dataclass(frozen=True)
class AuxiliaryParameters:
    sw_threshold: NDArray # The soil water threshold
    sw_exp: NDArray # The soil water exponent
    n_alpha: NDArray # The water table depth factor
    q_10: NDArray # The temperature factor range
    porosity: float # Porosity of this zone, must be in the range (0, 1)
    depth: float # Depth of this zone, in millimeters
    passive_water_storage: float # Passive water storage in this zone


    def soil_water_factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the dependence on the soil moisture
        """
        greater_mask: NDArray = forc.s_w >= self.sw_threshold
        less_mask: NDArray = ~greater_mask

        less: NDArray = (forc.s_w / self.sw_threshold) ** self.sw_exp
        greater: NDArray = ((1 - forc.s_w) / (1 - self.sw_threshold)) ** self.sw_exp

        return greater_mask * greater + less_mask * less # type: ignore
    

    def temperature_factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the dependence on temperature
        """
        return self.q_10 ** ((forc.hydro_forc.temp - 20.0) / 10.0) # type: ignore
    

    def water_table_factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the dependence on the water table
        """
        fzw_mask: NDArray = self.n_alpha == 0

        first_term: NDArray = np.ones_like(fzw_mask, dtype=np.float64)
        second_term: NDArray = np.exp(-abs(self.n_alpha) * forc.z_w ** (self.n_alpha / abs(self.n_alpha))) 

        return fzw_mask * first_term + ~fzw_mask * second_term # type: ignore
    

    def factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the auxiliary factor for each function
        """
        return self.soil_water_factor(forc) * self.temperature_factor(forc) * self.water_table_factor(forc) # type: ignore


class ReactionNetwork:

    def __init__(
            self,
            primary_aqueous: list[PrimaryAqueousSpecies],
            mineral: list[MineralSpecies],
            secondary: list[SecondarySpecies],
            mineral_kinetics: MineralKineticData,
            exchange: list[ExchangeReaction]
        ) -> None:
        self.primary_aqueous: list[PrimaryAqueousSpecies] = primary_aqueous
        self.mineral: list[MineralSpecies] = mineral
        self.secondary: list[SecondarySpecies] = secondary
        self.mineral_kinetics: MineralKineticData = mineral_kinetics
        self.exchange: list[ExchangeReaction] = exchange

        species_types: list[str] = ["primary"] * len(primary_aqueous) + ["secondary"] * len(secondary)
        names: list[str] = [x.name for x in primary_aqueous + secondary]
        if exchange:
            species_types += ["exchange"] * len(exchange)
            names += [x.name for x in exchange]
            names.insert(len(primary_aqueous), "X-")
            species_types.insert(len(primary_aqueous), "exchange")

        species_types += ["mineral"] * len(mineral)
        names += [x.name for x in mineral]

        self.__species: DataFrame = DataFrame({"name": names, "type": species_types}).set_index("name")


    @property
    def species_order(self) -> list[str]:
        return self.__species.index.tolist()


    @property
    def has_exchange(self) -> bool:
        """
        Boolean test for whether or not there are exchange species included in this reaction network
        """
        return bool(self.exchange)


    @property
    def charges(self) -> Series[float]:
        """
        Return a series of charges for all species in the model
        """
        charge_df: DataFrame = self.species.copy()
        charge_df["charge"] = 0.0
        for spec in self.primary_aqueous + self.secondary + self.exchange:
            charge_df.loc[spec.name, "charge"] = spec.charge

        charge_df.loc["X-", "charge"] = -1.0
        return charge_df["charge"][self.species_order]
    

    @property
    def species(self) -> DataFrame:
        """
        Get DataFrame of all species in a dataframe with 1 column called "type"
        """
        return self.__species
    

    @property
    def equilibrium_species(self) -> DataFrame:
        """
        Return a DataFrame with the species name as the index and only the column `type` on the
        species names. This includes primary, exchange, and secondary aqueous species
        """
        df: DataFrame = self.species
        return df.loc[df.type.isin(["primary", "exchange", "secondary"])].copy()


    @property
    def kinetic_species(self) -> DataFrame:
        """
        Get the matrix of only the aqueous species
        """
        return self.__species.loc[self.__species.type.isin(("primary", "mineral", "secondary"))].copy()

    
    @property
    def equilibrium_parameters(self) -> EquilibriumParameters:
        """
        Construct the equilibrium parameters from the database
        """
        # Construct the mass and charge conservation matrix
        mass_stoich_df: DataFrame = self.species
        for spec in self.secondary + self.exchange:
            mass_stoich_df[spec.name] = spec.stoichiometry

        total_species: list[str] = mass_stoich_df.loc[(mass_stoich_df.type.isin(("primary", "mineral"))) | (mass_stoich_df.index == "X-")].index.tolist()
        mass_stoich_df = mass_stoich_df.loc[total_species].drop(columns="type").fillna(0.0)
        primary_eye = DataFrame(np.eye(mass_stoich_df.shape[0]), columns=mass_stoich_df.index, index=mass_stoich_df.index)
        mass_stoich_df = pd.concat([primary_eye, mass_stoich_df], axis=1)

        rows = []
        for i, row in mass_stoich_df.iterrows():
            if i == "H+":
                # Use charge balance for mass balance on 'H+'
                new_row = self.charges.loc[mass_stoich_df.columns]
                new_row.name = "Charge"
                rows.append(new_row)

            else:
                new_row = row.abs()
                new_row.name = f"Tot_{new_row.name}"
                rows.append(new_row)

        mass_stoich_df = DataFrame(rows)[self.species_order]

        # Construct the stoichiometry matrix
        sec_stoich_df: DataFrame = self.species
        for spec in self.secondary + self.exchange:
            sec_stoich_df[spec.name] = spec.stoichiometry
        sec_stoich_df = sec_stoich_df.drop(columns=["type"]).fillna(0.0).T[self.species_order]

        sec_eq_vec: Series = Series(np.array([x.eq_consts[1] for x in self.secondary] + [x.log10_k_eq for x in self.exchange]), index=[x.name for x in self.secondary + self.exchange])

        return EquilibriumParameters(
            stoich=sec_stoich_df,
            equilibrium=sec_eq_vec,
            total=mass_stoich_df
        )


    @property
    def tst_params(self) -> TstParameters:
        """
        Construct the TST parameters for this reaction network
        """
        # Stoichiometry
        mineral_stoich_df: DataFrame = self.species
        for mineral in self.mineral:
            mineral_stoich_df[mineral.name] = mineral.stoichiometry

        mineral_stoich_df = mineral_stoich_df.drop(columns="type").fillna(0.0).T[self.species_order]

        # Dependence
        tst_dep_df: DataFrame = self.species
        for mineral in self.mineral:
            tst_dep_df[mineral.name] = 0.0

        for (name, reaction) in self.mineral_kinetics.tst_reactions.items():
            tst_dep_df[name] = reaction.dependence
        tst_dep_df = tst_dep_df.drop(columns="type").fillna(0.0).T[self.species_order]

        # Equilibrium constants
        min_eq_const: Series = Series([x.eq_consts[1] for x in self.mineral], index=[x.name for x in self.mineral])

        return TstParameters(
            stoich=mineral_stoich_df,
            dep=tst_dep_df,
            min_eq_const=min_eq_const
        )


    @property
    def monod_params(self) -> MonodParameters:
        """
        Construct the Monod parameters for this reaction network
        """
        monod_df: DataFrame = self.species
        for mineral in self.mineral:
            monod_df[mineral.name] = 0.0
        inhib_df = monod_df.copy()


        for (name, reaction) in self.mineral_kinetics.monod_reactions.items():
            monod_df[reaction.mineral_name] = reaction.monod_terms
            inhib_df[reaction.mineral_name] = reaction.inhib_terms

        monod_df: DataFrame = monod_df.drop(columns=["type"]).fillna(0.0).T[self.species_order]
        inhib_df: DataFrame = inhib_df.drop(columns=["type"]).fillna(0.0).T[self.species_order]

        return MonodParameters(
            monod_mat=monod_df,
            inhib_mat=inhib_df
        )


    @property
    def species_names(self) -> list[str]:
        """
        Return the names of the species, in order, that they are solved
        """
        return self.species_order

    @property
    def mineral_stoichiometry(self) -> DataFrame:
        """
        Return a dataframe of of the mineral stoichiometry
        """
        stoich_df: DataFrame = self.species

        for mineral in self.mineral:
            stoich_df[mineral.name] = mineral.stoichiometry

        return stoich_df.drop(columns="type").fillna(0.0).copy()
    
    @property
    def transport_mask(self) -> NDArray:
        """
        Get a boolean mask for the species that are either mobile or immobile. Mineral species 
        and exchange sites (X-) are immobile and will not be moved during transport
        """
        raise NotImplementedError()