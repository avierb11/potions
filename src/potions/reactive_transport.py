from __future__ import annotations
from abc import abstractmethod
from typing import Callable, Any, Final, TypeVar, Generic
from numpy.typing import NDArray
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.integrate import solve_ivp
import scipy.linalg as la
from scipy.optimize import fsolve
from numpy import float64 as f64

from .database import (
    ExchangeReaction,
    MineralKineticData,
    MineralKineticReaction,
    MineralSpecies,
    MonodReaction,
    PrimaryAqueousSpecies,
    SecondarySpecies,
    TstReaction,
)

from .interfaces import Zone, StepResult
from .common_types_compiled import HydroForcing
from .common_types import ChemicalState, RtForcing, Vector, Matrix, M, N
from .reaction_network import (
    MonodParameters,
    TstParameters,
    EquilibriumParameters,
    AuxiliaryParameters,
    MineralParameters,
)


@dataclass(frozen=True)
class MiscData:
    mineral_stoichiometry: NDArray
    species_mobility: NDArray


@dataclass(frozen=True)
class RtStep:
    """Holds the results of a single time step for a ReactiveTransportZone."""

    state: NDArray
    forc_flux: NDArray
    vap_flux: NDArray
    lat_flux: NDArray
    vert_flux: NDArray


class ReactiveTransportZone:
    """A zone that solves reactive transport

    This class acts as an ODE solver and orchestrator. It is initialized with
    functions that define the specific transport and kinetic reaction logic.
    This allows for a flexible definition of the model's biogeochemistry without
    hard-coding it into the zone itself.

    The `step` method uses operator splitting:
    1. Solves the ODE for transport and kinetic reactions.
    2. Solves for instantaneous chemical equilibrium.
    """

    def __init__(
        self,
        monod: MonodParameters,
        tst: TstParameters,
        eq: EquilibriumParameters,
        aux: AuxiliaryParameters,
        min: MineralParameters,
        misc: MiscData,
        name: str = "unnamed",
    ) -> None:
        """Initializes the zone with specific logic functions and parameters."""
        self.name: str = name
        self.monod: MonodParameters = monod
        self.tst: TstParameters = tst
        self.eq: EquilibriumParameters = eq
        self.aux: AuxiliaryParameters = aux
        self.min: MineralParameters = min
        self.misc: MiscData = misc

    def mass_balance_ode(self, chms: NDArray, d: RtForcing) -> NDArray:
        """Calculates the net rate of change of concentration (dC/dt).

        This method is conceptually based on the reactive transport equation, where the change in mass
        comes from the change from transports and the change due to transport:
        dm/dt = (dm/dt)_reaction + (dm/dt)_transport
        """
        reaction_rate_vec: NDArray = self.reaction_rate(
            chms, d
        )  # Rate of production or consumption of each of the mobile primary species (not minerals). Positive is production, negative is consumption
        transport_rate_vec: NDArray = self.transport_rate(
            chms, d
        )  # Rate of transport of of each of the mobile primary species
        return reaction_rate_vec + transport_rate_vec

    def reaction_rate(self, chms: NDArray, d: RtForcing) -> NDArray:
        """
        Calculate the rate of reaction for this time step - the rate that the primary species are produced or consumed
        """
        monod_rate: NDArray = self.monod.rate(
            chms
        )  # Reaction rate of each species from Monod Reactions
        tst_rate: NDArray = self.tst.rate(
            chms
        )  # Reaction rate of each species from TST-type reactions
        aux_rate: NDArray = self.aux.factor(
            d
        )  # Reaction rate modulation from soil moisture, temperature, and water table factors

        mineral_rates: NDArray = (
            self.min.rate_const
            * self.min.surface_area
            * aux_rate
            * (monod_rate + tst_rate)
        )

        all_species_rates = self.misc.mineral_stoichiometry @ mineral_rates

        return all_species_rates

    def transport_rate(self, chms: NDArray, d: RtForcing) -> NDArray:
        """
        Calculate the rate of transport for this time step for each of the aqueous species, including primary and secondary species
        This also includes the external transport into the zone.
        """
        mass_in: NDArray = d.hydro_forc.q_in * d.conc_in
        # mass_out: NDArray = (d.q_in / d.storage) * (d.conc_in - chms)
        mass_out: NDArray = d.q_out * chms

        mass_change: NDArray = mass_in - mass_out
        mass_change[~self.misc.species_mobility] = (
            0.0  # Set all immobile (mineral) species mass change to zero
        )

        return mass_change

    def step(
        self, c_0: NDArray, d: RtForcing, dt: float, q_in: float, debug: bool = False
    ) -> RtStep:
        """Advances the chemical state by one time step."""

        # 1. Solve the ODE for kinetic reactions and transport
        def ode(_t: float, c: NDArray) -> NDArray:
            return self.mass_balance_ode(c_0, d)

        # def residual(c: NDArray) -> NDArray:
        #     return (c_0 - c) + dt * self.mass_balance_ode(c, d)

        ode_res = solve_ivp(ode, [0, dt], y0=c_0)
        # opt_res = fsolve(residual, c_0)

        # Solve for the next concentration using some rootfinding method
        c_after_rt: NDArray = ode_res.y[:, -1]
        # c_after_rt: NDArray = opt_res

        if debug:
            print(f"C after Reaction+Transport: \n{c_after_rt}")

        # After finding the new concentrations that are not at equilibrium, solve for the equilibrium concentrations
        c_after_eq = self.eq.solve_equilibrium(c_after_rt)

        # 3. Calculate output fluxes for the time step
        forc_flux = q_in * d.conc_in
        vap_flux = np.zeros_like(c_after_eq)

        if debug:
            print(f"C after equilibrium: \n{c_after_eq}")
            tot_after_rt = self.eq.total.values @ c_after_rt
            tot_after_eq = self.eq.total.values @ c_after_eq
            print(f"Total before equilibrium: {tot_after_rt}")
            print(f"Total after equilibrium:  {tot_after_eq}")

        total_q_out_water = d.q_out
        if total_q_out_water > 1e-9:
            total_mass_out_flux = total_q_out_water * c_after_eq
            lat_flux = total_mass_out_flux * (d.q_lat_out / total_q_out_water)
            vert_flux = total_mass_out_flux * (d.q_vert_out / total_q_out_water)
        else:
            lat_flux = np.zeros_like(c_after_eq)
            vert_flux = np.zeros_like(c_after_eq)

        return RtStep(
            state=c_after_eq,
            forc_flux=forc_flux,
            vap_flux=vap_flux,
            lat_flux=lat_flux,
            vert_flux=vert_flux,
        )

    def columns(self, zone_id: int) -> list[str]:
        """Gets the column names for this zone for the output DataFrame."""
        # This will need to be updated based on the number of species.
        name = f"{self.name}_{zone_id}"
        return [
            f"c_{name}",
            f"j_forc_{name}",
            f"j_vap_{name}",
            f"j_lat_{name}",
            f"j_vert_{name}",
        ]
