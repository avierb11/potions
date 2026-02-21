from __future__ import annotations
from dataclasses import dataclass
from typing import Final
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from scipy.optimize import fsolve
import scipy.linalg as la
from numpy.typing import NDArray

from .utils import find_root_multi
from .database import (
    ExchangeReaction,
    MineralKineticData,
    MineralSpecies,
    PrimaryAqueousSpecies,
    SecondarySpecies,
)
from .common_types import Vector, Matrix, ChemicalState, RtForcing


@dataclass(frozen=True)
class MonodParameters:
    """
    A class containing the Monod parameters for a reaction network in a reactive transport model.

    This class holds two matrices: `monod_mat` and `inhib_mat`, which are used to calculate the
    reaction rate using Monod kinetics. These matrices define how the water chemistry alters the
    reaction rate of solids in the model.

    The matrices contain values for all species involved in the reaction network, including primary,
    secondary (controlled by equilibrium), and mineral species. This allows the model to track all
    chemical states in a single vector, rather than managing them separately.

    The columns of both matrices represent all species in the reaction network in the order:
    Primary species, Secondary species (controlled by equilibrium), and Mineral species.

    Important Notes:
        - Values in both `monod_mat` and `inhib_mat` are only non-NaN if they affect the reaction rate.
          This ensures that only relevant species are considered in the calculation.
        - None of the values in either matrix can be zero. A zero value would cause numerical instability
          during the calculation of the reaction rate, leading to incorrect or undefined results.

    Attributes:
        monod_mat (DataFrame): A matrix of shape (number of minerals x number of total species)
                               representing the Monod kinetic parameters for each mineral and species.
                               The columns represent all species in the reaction network in the order:
                               Primary species, Secondary species (controlled by equilibrium),
                               and Mineral species.

        inhib_mat (DataFrame): A matrix of shape (number of minerals x number of total species)
                               representing the inhibition parameters for each mineral and species.
                               The columns follow the same order as `monod_mat`.
    """

    monod_mat: DataFrame
    inhib_mat: DataFrame

    def rate(self, chms: NDArray) -> NDArray:
        """
        Calculate the rate of reaction using Monod kinetics based on the concentration of species.

        This method computes the reaction rate for each mineral in the network, taking into account
        the concentrations of all species involved in the reaction. The rate is a value between 0 and 1,
        which modulates the maximum rate of the reaction. This function is essential for modeling how
        the chemical state of the system influences the kinetics of mineral reactions.

        Parameters:
            conc (NDArray): A 1D numpy array of size equal to the number of species in the reaction
                            network. The order of species in this array must match the columns of
                            `monod_mat` and `inhib_mat`. Each entry in the array represents the
                            concentration of a species in the system.

        Returns:
            NDArray: A 1D numpy array of size equal to the number of minerals in the network.
                     Each entry corresponds to the reaction rate for a specific mineral, which is
                     modulated by the chemical state of the system.

        Example:
            >>> import numpy as np
            >>> params = MonodParameters()
            >>> conc = np.array([1.0, 2.0, 0.5])  # Concentrations of primary, secondary, and mineral species
            >>> rates = params.rate(conc)
            >>> print(rates)
            [0.6, 0.4]  # Example output: reaction rates for two minerals
        """
        monod = np.zeros(self.monod_mat.shape[0], dtype=np.float64)
        inhib = np.zeros(self.inhib_mat.shape[0], dtype=np.float64)

        for i, row in enumerate(self.monod_mat.values):
            monod[i] = np.nanprod(chms / (row + chms))

        for i, row in enumerate(self.inhib_mat.values):
            inhib[i] = np.nanprod(row / (row + chms))

        return monod * inhib


@dataclass(frozen=True)
class TstParameters:
    """
    The Transition-State Theory (TST) parameters, containing two matrices of shape
    (number of minerals x number of total species in the simulation).

    This class is used to define the parameters for a reaction network based on Transition-State Theory
    (TST), which is commonly used to model the dissolution kinetics of minerals in reactive transport models.
    The class contains matrices that define the stoichiometry and dependencies of the reactions, as well
    as equilibrium constants for the minerals involved.

    The matrices contain values for all species involved in the reaction network, including primary,
    secondary (controlled by equilibrium), and mineral species. This allows the model to track all
    chemical states in a single vector, rather than managing them separately.

    The columns of both matrices represent all species in the reaction network in the order:
    Primary species, Secondary species (controlled by equilibrium), and Mineral species.

    Important Notes:
        - Values in both `stoich` and `dep` matrices are only non-NaN if they affect the reaction rate.
          This ensures that only relevant species are considered in the calculation.
        - The `min_eq_const` vector contains equilibrium constants for the minerals, which are used
          to calculate the saturation index and determine the direction of the reaction.

    Attributes:
        stoich (DataFrame): A matrix of shape (number of minerals x number of total species)
                           representing the stoichiometric coefficients for each mineral and species.
                           The columns represent all species in the reaction network in the order:
                           Primary species, Secondary species (controlled by equilibrium),
                           and Mineral species.

        dep (DataFrame): A matrix of shape (number of minerals x number of total species)
                        representing the dependency coefficients for each mineral and species.
                        The columns follow the same order as `stoich`.

        min_eq_const (Series): A 1D vector of equilibrium constants for each mineral in the network.
                              These constants are used to determine the saturation index and the
                              direction of the reaction.
    """

    stoich: DataFrame  # Stoichiometry of the mineral reactions
    dep: DataFrame  # Dependence of the rate law on other species concentrations (typically pH through H+ or OH- dependence)
    min_eq_const: Series  # Vector of equilibrium constants

    def rate(self, chms: NDArray) -> NDArray:
        """
        Calculate the rate of reaction using Transition-State Theory (TST) kinetics.

        This method computes the reaction rate for each mineral in the network, based on the
        concentrations of all species involved in the reaction. The rate is determined by the
        difference between the actual activity product (IAP) and the equilibrium constant (K),
        and is used to model dissolution-only reactions for minerals.

        Parameters:
            all_species_conc (NDArray): A 1D numpy array of size equal to the number of species
                                        in the reaction network. The order of species in this array
                                        must match the columns of `stoich` and `dep`. Each entry
                                        represents the concentration of a species in the system.

        Returns:
            NDArray: A 1D numpy array of size equal to the number of minerals in the network.
                        Each entry corresponds to the reaction rate for a specific mineral, based on
                        the difference between the actual activity product and the equilibrium constant.

        Example:
            >>> import numpy as np
            >>> params = TstParameters()
            >>> conc = np.array([1.0, 2.0, 0.5])  # Concentrations of primary, secondary, and mineral species
            >>> rates = params.rate(conc)
            >>> print(rates)
            [0.2, 0.1]  # Example output: reaction rates for two minerals
        """
        log_prim: NDArray = np.log10(chms)

        log_dep: NDArray = self.dep.values @ log_prim
        log_iap: NDArray = self.stoich.values @ log_prim

        dep: NDArray = 10**log_dep
        iap: NDArray = 10**log_iap

        return dep * (1.0 - iap / self.min_eq_const)


@dataclass(frozen=True)
class EquilibriumParameters:
    stoich: DataFrame  # Matrix describing the stoichiometry of the secondary species
    log_eq_consts: Series  # Vector of the equilibrium constants for the secondary species in log10 scale
    total: DataFrame  # Matrix describing the mass and charge balance of the species

    @staticmethod
    def from_species(
        species: Series,
        primary: list[PrimaryAqueousSpecies],
        secondary: list[SecondarySpecies],
        exchange: list[ExchangeReaction],
    ) -> EquilibriumParameters:
        """
        Construct the equilibrium parameters from the database
        """
        raise NotImplementedError()

    @property
    def stoich_null_space(self) -> Matrix:
        """Return the null space of the stoichiometry matrix"""
        return la.null_space(self.stoich)  # type: ignore

    @property
    def log10_k_w(self) -> Vector:
        """Return a vector of the equilibrium constants in base-10 logarithm"""
        return self.log_eq_consts.to_numpy()

    @property
    def x_particular(self) -> Vector:
        """
        Return a vector of the particular solution of the null space of the stoichiometry
        """
        return la.pinv(self.stoich) @ self.log10_k_w  # type: ignore

    def solve_equilibrium(self, chms: NDArray, debug: bool = False) -> NDArray:
        """
        Solve for the equilibrium concentrations of all of the species
        """
        c_tot: Final[NDArray] = (
            self.total.values @ chms
        )  # Total concentrations in the system
        if debug:
            print(f"{c_tot=}")

        def conc(tot_conc: NDArray) -> NDArray:
            """
            Return a vector of the aqueous concentrations in base-10 logarithm
            """
            return 10 ** (self.stoich_null_space @ tot_conc + self.x_particular)

        def f(tot_conc: NDArray) -> NDArray:
            return c_tot - self.total.values @ conc(tot_conc)

        # sol: NDArray = fsolve(f, x0=c_tot)
        sol: NDArray = find_root_multi(f=f, x_0=c_tot, debug=debug)

        new_conc: NDArray = conc(sol)

        return new_conc


class MineralParameters:
    def __init__(
        self,
        volume_fraction: NDArray,
        ssa: NDArray,
        rate_const: NDArray,
    ) -> None:
        self.volume_fraction: NDArray = (
            volume_fraction  # Volume fraction of each mineral species in grams mineral/grams solid
        )
        self.ssa: NDArray = (
            ssa  # Specific surface area in units of m^2 mineral surface area / grams mineral
        )
        self.rate_const: NDArray = rate_const  # Rate constant for the reactions

    @property
    def surface_area(self) -> NDArray:
        return self.volume_fraction * self.ssa  # type: ignore


@dataclass(frozen=True)
class MineralAuxParams:
    sw_threshold: float
    sw_exp: float
    n_alpha: float
    q_10: float


@dataclass(frozen=True)
class ZoneDimensions:
    porosity: float
    depth: float
    passive_water_storage: float


@dataclass(frozen=True)
class AuxiliaryParameters:
    sw_threshold: NDArray  # The soil water threshold
    sw_exp: NDArray  # The soil water exponent
    n_alpha: NDArray  # The water table depth factor
    q_10: NDArray  # The temperature factor range
    porosity: float  # Porosity of this zone, must be in the range (0, 1)
    depth: float  # Depth of this zone, in millimeters
    passive_water_storage: float  # Passive water storage in this zone

    @staticmethod
    def from_minerals(
        min_params: list[MineralAuxParams], zone_params: ZoneDimensions
    ) -> AuxiliaryParameters:
        sw_thresholds: np.ndarray = np.array(
            list(map(lambda x: x.sw_threshold, min_params))
        )
        sw_exps: np.ndarray = np.array(list(map(lambda x: x.sw_exp, min_params)))
        n_alphas: np.ndarray = np.array(list(map(lambda x: x.n_alpha, min_params)))
        q_10s: np.ndarray = np.array(list(map(lambda x: x.q_10, min_params)))

        return AuxiliaryParameters(
            sw_threshold=sw_thresholds,
            sw_exp=sw_exps,
            n_alpha=n_alphas,
            q_10=q_10s,
            porosity=zone_params.porosity,
            depth=zone_params.depth,
            passive_water_storage=zone_params.passive_water_storage,
        )

    def soil_water_factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the dependence on the soil moisture
        """
        greater_mask: NDArray = forc.s_w >= self.sw_threshold
        less_mask: NDArray = ~greater_mask

        less: NDArray = (forc.s_w / self.sw_threshold) ** self.sw_exp
        greater: NDArray = ((1 - forc.s_w) / (1 - self.sw_threshold)) ** self.sw_exp

        arr: NDArray = greater_mask * greater + less_mask * less

        return arr

    def temperature_factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the dependence on temperature
        """
        return self.q_10 ** ((forc.hydro_forc.temp - 20.0) / 10.0)  # type: ignore

    def water_table_factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the dependence on the water table
        """
        fzw_mask: NDArray = self.n_alpha == 0

        n_alpha_vals: NDArray = np.ones_like(fzw_mask, dtype=np.float64)
        second_term: NDArray = np.exp(
            -abs(self.n_alpha) * forc.z_w ** (self.n_alpha / abs(self.n_alpha))
        )

        n_alpha_vals[~fzw_mask] = second_term

        return n_alpha_vals

    def factor(self, forc: RtForcing) -> Vector:
        """
        Calculate the auxiliary factor for each function
        """
        return self.soil_water_factor(forc) * self.temperature_factor(forc) * self.water_table_factor(forc)  # type: ignore


class ReactionNetwork:

    def __init__(
        self,
        primary_aqueous: list[PrimaryAqueousSpecies],
        mineral: list[MineralSpecies],
        secondary: list[SecondarySpecies],
        mineral_kinetics: MineralKineticData,
        exchange: list[ExchangeReaction],
    ) -> None:
        self.primary_aqueous: list[PrimaryAqueousSpecies] = primary_aqueous
        self.mineral: list[MineralSpecies] = mineral
        self.secondary: list[SecondarySpecies] = secondary
        self.mineral_kinetics: MineralKineticData = mineral_kinetics
        self.exchange: list[ExchangeReaction] = exchange

        species_types: list[str] = ["primary"] * len(primary_aqueous) + [
            "secondary"
        ] * len(secondary)
        names: list[str] = [x.name for x in primary_aqueous + secondary]
        if exchange:
            species_types += ["exchange"] * len(exchange)
            names += [x.name for x in exchange]
            names.insert(len(primary_aqueous), "X-")
            species_types.insert(len(primary_aqueous), "exchange")

        species_types += ["mineral"] * len(mineral)
        names += [x.name for x in mineral]

        self.__species: DataFrame = DataFrame(
            {"name": names, "type": species_types}
        ).set_index("name")

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
        return self.__species[["type"]]

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
        return self.__species.loc[
            self.__species.type.isin(("primary", "mineral", "secondary"))
        ].copy()

    @property
    def equilibrium_parameters(self) -> EquilibriumParameters:
        """
        Construct the equilibrium parameters from the database
        """
        # Construct the mass and charge conservation matrix
        mass_stoich_df: DataFrame = self.species
        for spec in self.secondary + self.exchange:
            mass_stoich_df[spec.name] = spec.stoichiometry

        total_species: list[str] = mass_stoich_df.loc[
            (mass_stoich_df.type.isin(("primary", "mineral")))
            | (mass_stoich_df.index == "X-")
        ].index.tolist()
        mass_stoich_df = (
            mass_stoich_df.loc[total_species].drop(columns="type").fillna(0.0)
        )
        primary_eye = DataFrame(
            np.eye(mass_stoich_df.shape[0]),
            columns=mass_stoich_df.index,
            index=mass_stoich_df.index,
        )
        mass_stoich_df = pd.concat([primary_eye, mass_stoich_df], axis=1)

        rows = []
        for i, row in mass_stoich_df.iterrows():
            if i == "H+":
                # Use charge balance for mass balance on 'H+'
                new_row = self.charges.loc[mass_stoich_df.columns]
                new_row.name = "Charge"
                rows.append(new_row.to_frame().T)

            else:
                new_row = row.abs()
                new_row.name = f"Tot_{new_row.name}"
                rows.append(new_row.to_frame().T)

        mass_stoich_df = pd.concat(rows)[self.species_order]

        # Construct the stoichiometry matrix
        sec_stoich_df: DataFrame = self.species
        for spec in self.secondary + self.exchange:
            sec_stoich_df[spec.name] = spec.stoichiometry
        sec_stoich_df = (
            sec_stoich_df.drop(columns=["type"]).fillna(0.0).T[self.species_order]
        )

        sec_eq_vec: Series = Series(
            np.array(
                [x.eq_consts[1] for x in self.secondary]
                + [x.log10_k_eq for x in self.exchange]
            ),
            index=[x.name for x in self.secondary + self.exchange],
        )

        return EquilibriumParameters(
            stoich=sec_stoich_df, log_eq_consts=sec_eq_vec, total=mass_stoich_df
        )

    @property
    def tst_params(self) -> TstParameters:
        """
        Construct the TST parameters for this reaction network
        """
        # Stoichiometry
        mineral_stoich_df: DataFrame = self.species
        for mineral in self.mineral:
            if mineral.name in self.mineral_kinetics.tst_reactions:
                mineral_stoich_df[mineral.name] = mineral.stoichiometry
            else:
                mineral_stoich_df[mineral.name] = 0.0

        mineral_stoich_df = (
            mineral_stoich_df.drop(columns="type").fillna(0.0).T[self.species_order]
        )

        # Dependence
        tst_dep_df: DataFrame = self.species
        for mineral in self.mineral:
            tst_dep_df[mineral.name] = 0.0

        for name, reaction in self.mineral_kinetics.tst_reactions.items():
            tst_dep_df[name] = reaction.dependence
        tst_dep_df = tst_dep_df.drop(columns="type").fillna(0.0).T[self.species_order]

        # Equilibrium constants

        min_eq_const: Series = Series(
            [
                x.eq_consts[1] if x.name in self.mineral_kinetics.tst_reactions else 1.0
                for x in self.mineral
            ],
            index=[x.name for x in self.mineral],
        ).astype(float)

        return TstParameters(
            stoich=mineral_stoich_df, dep=tst_dep_df, min_eq_const=min_eq_const
        )

    @property
    def monod_params(self) -> MonodParameters:
        """
        Construct the Monod parameters for this reaction network
        """
        monod_df: DataFrame = self.species
        for mineral in self.mineral:
            monod_df[mineral.name] = 0.0
        inhib_df: DataFrame = monod_df.copy()

        for name, reaction in self.mineral_kinetics.monod_reactions.items():
            monod_df[reaction.mineral_name] = reaction.monod_terms
            inhib_df[reaction.mineral_name] = reaction.inhib_terms

        monod_df = monod_df.drop(columns=["type"]).copy().T[self.species_order]
        inhib_df = (
            inhib_df.drop(columns=["type"]).copy().astype(float).T[self.species_order]
        )

        return MonodParameters(monod_mat=monod_df, inhib_mat=inhib_df)

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

        mineral_names: list[str] = []
        for mineral in self.mineral:
            stoich_df[mineral.name] = mineral.stoichiometry
            mineral_names.append(mineral.name)

        return stoich_df[mineral_names].copy().fillna(0.0)

    @property
    def transport_mask(self) -> NDArray:
        """
        Get a boolean mask for the species that are either mobile or immobile. Mineral species
        and exchange sites (X-) are immobile and will not be moved during transport
        """
        mobile_vals = self.species["type"] != "mineral"
        return mobile_vals.to_numpy()
