from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series


from .database import (
    MineralKineticData,
    MineralSpecies,
    PrimaryAqueousSpecies,
    SecondarySpecies,
)

# from potions.core import
# from .kinetic_structures import EquilibriumParameters, TstParameters, MonodParameters

from .kinetic_structures import (
    EquilibriumParameters,
    TstParameters,
    MonodParameters,
    PARAMETERS_PER_MINERAL,
)


class ReactionNetwork:

    def __init__(
        self,
        primary_aqueous: list[PrimaryAqueousSpecies],
        mineral: list[MineralSpecies],
        secondary: list[SecondarySpecies],
        mineral_kinetics: MineralKineticData,
        # exchange: list[ExchangeReaction],
    ) -> None:
        self.primary_aqueous: list[PrimaryAqueousSpecies] = primary_aqueous
        self.mineral: list[MineralSpecies] = mineral
        self.secondary: list[SecondarySpecies] = secondary
        self.mineral_kinetics: MineralKineticData = mineral_kinetics
        # self.exchange: list[ExchangeReaction] = exchange

        species_types: list[str] = ["primary"] * len(primary_aqueous) + [
            "secondary"
        ] * len(secondary)
        names: list[str] = [x.name for x in primary_aqueous + secondary]
        # if exchange:
        #     species_types += ["exchange"] * len(exchange)
        #     names += [x.name for x in exchange]
        #     names.insert(len(primary_aqueous), "X-")
        #     species_types.insert(len(primary_aqueous), "exchange")

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
        # return bool(self.exchange)
        return False

    @property
    def charges(self) -> Series[float]:
        """
        Return a series of charges for all species in the model
        """
        charge_df: DataFrame = self.species.copy()
        charge_df["charge"] = 0.0
        # for spec in self.primary_aqueous + self.secondary + self.exchange:
        for spec in self.primary_aqueous + self.secondary:
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
        # for spec in self.secondary + self.exchange:
        for spec in self.secondary:
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
        # for spec in self.secondary + self.exchange:
        for spec in self.secondary:
            sec_stoich_df[spec.name] = spec.stoichiometry
        sec_stoich_df = (
            sec_stoich_df.drop(columns=["type"]).fillna(0.0).T[self.species_order]
        )

        sec_eq_vec: Series = Series(
            np.array(
                [x.eq_consts[1] for x in self.secondary]
                # + [x.log10_k_eq for x in self.exchange]
            ),
            # index=[x.name for x in self.secondary + self.exchange],
            index=[x.name for x in self.secondary],
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

        for _name, reaction in self.mineral_kinetics.monod_reactions.items():
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
    def mineral_species_names(self) -> list[str]:
        """
        Return the names of the species, in order, that they are solved
        """
        return [m.name for m in self.mineral]

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

    @property
    def mineral_molar_masses(self) -> NDArray:
        """The molar masses of each of the minerals"""
        return np.array([x.molar_mass for x in self.mineral])

    @property
    def rate_consts(self) -> NDArray:
        log10_rate_consts: list[float] = []
        for m in self.mineral:
            if m.name in self.mineral_kinetics.monod_reactions:
                log10_rate_consts.append(
                    self.mineral_kinetics.monod_reactions[m.name].rate_constant
                )
            else:
                log10_rate_consts.append(
                    self.mineral_kinetics.tst_reactions[m.name].rate_constant
                )
        return 10.0 ** np.array(log10_rate_consts)

    @property
    def num_minerals(self) -> int:
        return self.monod_params.monod_mat.shape[0]

    @property
    def num_mineral_parameters(self) -> int:
        """The number of mineral parameters"""
        return PARAMETERS_PER_MINERAL * self.num_minerals

    @property
    def num_aqueous_species(self) -> int:
        """Get the total number of aqueous species, equal to the number of primary aqueous and secondary species"""
        return self.species["type"].isin(["primary", "secondary"]).sum()

    def get_default_aqueous_initial_state(self, init_conc: float = 1e-6) -> NDArray:
        """Get an array of the initial concentrations for this zone"""
        return np.full(
            shape=(self.num_aqueous_species,), fill_value=init_conc, dtype=np.float64
        )
