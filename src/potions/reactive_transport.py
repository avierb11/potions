from __future__ import annotations
from abc import abstractmethod
from typing import Callable, Any, Final, TypeVar, Generic
from numpy.typing import NDArray
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
import scipy.linalg as la
from scipy.optimize import fsolve
from numpy import float64 as f64

from .interfaces import Zone, StepResult
from .hydro import HydroForcing


M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
Vector = np.ndarray[tuple[N], np.dtype[f64]]
Matrix = np.ndarray[tuple[M, N], np.dtype[f64]]
# NumTot = TypeVar("NumTot", bound=int)  # Number of total species
# NumPrim = TypeVar("NumPrim", bound=int)  # Number of primary aqueous species
# NumMin = TypeVar("NumMin", bound=int)  # Number of mineral species
# NumSec = TypeVar("NumSec", bound=int)  # Number of secondary aqueous species
# NumSpec = TypeVar("NumSpec", bound=int)  # Number of species in the model
# NumAqueous = TypeVar("NumAqueous", bound=int)  # Number of aqueous species in the model



@dataclass(frozen=True)
class ChemicalState:
    """Represents the chemical state of a zone, partitioned by species type."""
    prim_aq_conc: Vector
    min_conc: Vector
    sec_conc: Vector

    def to_primary_array(self) -> Vector:
        """Concatenates primary aqueous and mineral species into a single array."""
        return np.concatenate([self.prim_aq_conc, self.min_conc])  # type: ignore
    
    @property
    def aqueous_concentrations(self) -> Vector:
        """
        Get a vector of aqueous concentrations, including primary and secondary
        """
        return np.concatenate([self.prim_aq_conc, self.sec_conc])  # type: ignore


@dataclass(frozen=True)
class RtForcing:
    """Contains the hydrologic and chemical drivers for a reactive transport step."""
    conc_in: ChemicalState
    q_in: float
    q_lat_out: float
    q_vert_out: float
    hydro_forc: HydroForcing
    storage: float # Water storage in the zone, in millimeters
    s_w: float # Fraction of soil taken up by water, ranges from [0,1], with 1 indicating all porosity is filled
    z_w: float # Depth of the water table

    @property
    def q_out(self) -> float:
        """The total flux of water out of this zone"""
        return self.q_lat_out + self.q_vert_out


@dataclass(frozen=True)
class MonodParameters:
    monod_mat: Matrix
    inhib_mat: Matrix

    def rate(self, prim_conc: Vector) -> Vector:
        """
        Calculate the rate of reaction using Monod kinetics
        """
        monod: Vector = (prim_conc / (self.monod_mat + prim_conc)).sum(axis=1)
        inhib: Vector = ((self.inhib_mat + prim_conc) / prim_conc).sum(axis=1)

        return monod * inhib # type: ignore
    

@dataclass(frozen=True)
class TstParameters:
    stoich: Matrix
    dep: Matrix
    min_eq_const: Vector # Vector of equilibrium constants

    def rate(self, prim_conc: Vector) -> Vector:
        """
        Calculate the rate of reaction using Monod kinetics
        """
        log_prim: NDArray = np.log10(prim_conc)

        log_dep: NDArray = self.dep @ log_prim # type: ignore
        log_iap: NDArray = self.stoich @ log_prim # type: ignore

        dep: NDArray = 10 ** log_dep # type: ignore
        iap: NDArray = 10 ** log_iap

        return dep * (1.0 - iap / self.min_eq_const) # type: ignore


@dataclass(frozen=True)
class EquilibriumParameters:
    stoich: Matrix # Matrix describing the stoichiometry of the secondary species
    equilibrium: Vector # Vector of the equilibrium constants for the secondary species
    total: Matrix # Matrix describing the mass and charge balance of the species 

    @property
    def stoich_null_space(self) -> Matrix:
        """Return the null space of the stoichiometry matrix
        """
        return la.null_space(self.stoich) # type: ignore
    
    @property
    def log10_k_w(self) -> Vector:
        """Return a vector of the equilibrium constants in base-10 logarithm
        """
        return np.log10(self.equilibrium) # type: ignore


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
    def __init__(self, volume_fraction: Vector, ssa: Vector, rate_const: Vector) -> None:
        self.volume_fraction: Vector = volume_fraction # Volume fraction of each mineral species in grams mineral/grams solid
        self.ssa: Vector = ssa # Specific surface area in units of m^2 mineral surface area / grams mineral
        self.rate_const: Vector = rate_const # Rate constant for the reactions


    @property
    def surface_area(self) -> Vector:
        return self.volume_fraction * self.ssa # type: ignore


@dataclass(frozen=True)
class AuxiliaryParameters:
    sw_threshold: Vector # The soil water threshold
    sw_exp: Vector # The soil water exponent
    n_alpha: Vector # The water table depth factor
    q_10: Vector # The temperature factor range
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

    
@dataclass(frozen=True)
class RtStep(StepResult[NDArray]):
    """Holds the results of a single time step for a ReactiveTransportZone."""
    state: Vector
    forc_flux: Vector
    vap_flux: Vector
    lat_flux: Vector
    vert_flux: Vector


class ReactiveTransportZone(Zone[NDArray, RtForcing, RtStep]):
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
      

    def _mass_balance_ode(self, chms: ChemicalState, d: RtForcing) -> NDArray:
        """Calculates the net rate of change of concentration (dC/dt).

        This method is conceptually based on the equation from the documentation:
        dm/dt = (dm/dt)_reaction + (dm/dt)_transport
        It calls the external functions provided during initialization.
        """
        reaction_rate_vec: Vector = self.reaction_rate(chms, d) # Rate of production or consumption of each of the mobile primary species (not minerals)
        transport_rate_vec: Vector = self.transport_rate(chms, d) # Rate of transport of of each of the mobile primary species 
        return reaction_rate_vec + transport_rate_vec


    def reaction_rate(self, chms: ChemicalState, d: RtForcing) -> Vector:
        """
        Calculate the rate of reaction for this time step - the rate that the primary species are produced or consumed
        """
        monod_rate: Vector = self.monod.rate(chms.prim_aq_conc)
        tst_rate: Vector = self.tst.rate(chms.prim_aq_conc)
        aux_rate: Vector = self.aux.factor(d)

        return self.min.rate_const * self.min.surface_area * aux_rate * (monod_rate + tst_rate) # type: ignore
    
    
    def transport_rate(self, chms: ChemicalState, d: RtForcing) -> Vector:
        """
        Calculate the rate of transport for this time step for each of the aqueous species, including primary and secondary species
        """
        transport_factor: float = d.q_in / d.storage
        tot_conc: Vector[NumTot] = self.eq.total @ chms.to_primary_array() # type: ignore

        return d.conc_in - transport_factor * tot_conc # type: ignore


    def step(self, s_0: NDArray, d: RtForcing, dt: float, q_in: NDArray) -> RtStep:
        """Advances the chemical state by one time step."""

        # 1. Solve the ODE for kinetic reactions and transport
        def f(t: float, c: NDArray) -> NDArray:
            return self._mass_balance_ode(c, d, q_in)

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
