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
from .hydro import HydroForcing
from .common_types import ChemicalState, RtForcing, Vector, Matrix, M, N
from .reaction_network import (
    MonodParameters,
    TstParameters,
    EquilibriumParameters,
    AuxiliaryParameters,
    MineralParameters,
)


@dataclass(frozen=True)
class RtStep(StepResult[NDArray]):
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
        name: str = "unnamed",
    ) -> None:
        """Initializes the zone with specific logic functions and parameters.

        Args:
            reaction_fn: A function calculating dC/dt from reactions.
            transport_fn: A function calculating dC/dt from transport.
            equilibrium_fn: A function that solves for equilibrium concentrations.
            params: A dictionary of parameters for the provided functions.
            name: The name of this zone type.
        """
        self.name: str = name
        self.monod: MonodParameters = monod
        self.tst: TstParameters = tst
        self.eq: EquilibriumParameters = eq
        self.aux: AuxiliaryParameters = aux
        self.min: MineralParameters = min

    def mass_balance_ode(self, chms: NDArray, d: RtForcing) -> NDArray:
        """Calculates the net rate of change of concentration (dC/dt).

        This method is conceptually based on the equation from the documentation:
        dm/dt = (dm/dt)_reaction + (dm/dt)_transport
        It calls the external functions provided during initialization.
        """
        reaction_rate_vec: NDArray = self.reaction_rate(
            chms, d
        )  # Rate of production or consumption of each of the mobile primary species (not minerals)
        transport_rate_vec: NDArray = self.transport_rate(
            chms, d
        )  # Rate of transport of of each of the mobile primary species
        return reaction_rate_vec + transport_rate_vec

    def reaction_rate(self, chms: NDArray, d: RtForcing) -> NDArray:
        """
        Calculate the rate of reaction for this time step - the rate that the primary species are produced or consumed
        """
        monod_rate: NDArray = self.monod.rate(chms)
        tst_rate: NDArray = self.tst.rate(chms)
        aux_rate: NDArray = self.aux.factor(d)

        return (
            self.min.rate_const
            * self.min.surface_area
            * aux_rate
            * (monod_rate + tst_rate)
        )

    def transport_rate(self, chms: NDArray, d: RtForcing) -> NDArray:
        """
        Calculate the rate of transport for this time step for each of the aqueous species, including primary and secondary species
        """
        return (d.q_in / d.storage) * (d.conc_in - chms)

    def step(self, s_0: NDArray, d: RtForcing, dt: float, q_in: NDArray) -> RtStep:
        """Advances the chemical state by one time step."""

        # 1. Solve the ODE for kinetic reactions and transport
        def f(t: float, c: NDArray) -> NDArray:
            return self.mass_balance_ode(c, d)

        res = solve_ivp(f, (0, dt), y0=s_0, dense_output=True)
        c_after_kinetics = res.y[:, -1]

        # 2. Solve for instantaneous chemical equilibrium
        c_new = self.eq.solve_equilibrium(c_after_kinetics)

        # 3. Calculate output fluxes for the time step
        forc_flux = q_in
        vap_flux = np.zeros_like(c_new)

        total_q_out_water = d.q_out
        if total_q_out_water > 1e-9:
            c_avg = res.sol(dt / 2)
            total_mass_out_flux = total_q_out_water * c_avg
            lat_flux = total_mass_out_flux * (d.q_lat_out / total_q_out_water)
            vert_flux = total_mass_out_flux * (d.q_vert_out / total_q_out_water)
        else:
            lat_flux = np.zeros_like(c_new)
            vert_flux = np.zeros_like(c_new)

        # Need to solve for equilibrium now

        return RtStep(
            state=c_new,
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
