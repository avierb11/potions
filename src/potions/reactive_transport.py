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
NumTot = TypeVar("NumTot", bound=int)  # Number of total species
NumPrim = TypeVar("NumPrim", bound=int)  # Number of primary aqueous species
NumMin = TypeVar("NumMin", bound=int)  # Number of mineral species
NumSec = TypeVar("NumSec", bound=int)  # Number of secondary aqueous species
NumSpec = TypeVar("NumSpec", bound=int)  # Number of species in the model
NumAqueous = TypeVar("NumAqueous", bound=int)  # Number of aqueous species in the model



@dataclass(frozen=True)
class ChemicalState(Generic[NumMin, NumPrim, NumTot, NumSec]):
    """Represents the chemical state of a zone, partitioned by species type."""
    prim_aq_conc: Vector[NumPrim]
    min_conc: Vector[NumMin]
    sec_conc: Vector[NumSec]

    def to_primary_array(self) -> Vector[NumTot]:
        """Concatenates primary aqueous and mineral species into a single array."""
        return np.concatenate([self.prim_aq_conc, self.min_conc])  # type: ignore


@dataclass(frozen=True)
class RtForcing:
    """Contains the hydrologic and chemical drivers for a reactive transport step."""
    conc_in: ChemicalState
    q_in: float
    q_lat_out: float
    q_vert_out: float
    hydro_forc: HydroForcing
    s_w: float
    z_w: float

    @property
    def q_out(self) -> float:
        """The total flux of water out of this zone"""
        return self.q_lat_out + self.q_vert_out


@dataclass(frozen=True)
class EquilibriumParameters(Generic[NumSec, NumTot, NumAqueous]):
    stoich: Matrix # Matrix describing the stoichiometry of the secondary species
    equilibrium: Vector[NumSec] # Vector of the equilibrium constants for the secondary species
    total: Matrix # Matrix describing the mass and charge balance of the species 

    @property
    def stoich_null_space(self) -> Matrix[NumAqueous, NumTot]:
        """Return the null space of the stoichiometry matrix
        """
        raise NotImplementedError()
    
    @property
    def log10_k_w(self) -> Vector[NumSec]:
        """Return a vector of the equilibrium constants in base-10 logarithm
        """
        return np.log10(self.equilibrium) # type: ignore


    @property
    def x_particular(self) -> Vector[NumAqueous]:
        """
        Return a vector of the particular solution of the null space of the stoichiometry
        """
        raise NotImplementedError()

    def solve_equilibrium(self, chms: ChemicalState) -> ChemicalState:
        """
        Solve for the equilibrium concentrations of all of the species
        """
        c_tot: Final[Vector[NumTot]] = self.total @ np.concatenate([chms.prim_aq_conc, chms.sec_conc]) # type: ignore


        def conc(x: Vector[NumTot]) -> Vector[NumAqueous]:
            """
            Return a vector of the aqueous concentrations in base-10 logarithm
            """
            return 10 ** (self.stoich_null_space @ x + self.x_particular) # type: ignore

        def f(x: Vector[NumTot]) -> Vector[NumTot]:
            return c_tot - self.total @ conc(x) # type: ignore

        sol = fsolve(f, x0=np.zeros_like(c_tot))

        new_conc: Vector[NumAqueous] = conc(sol) # type: ignore
        

        prim_conc: Vector[NumTot]
        sec_conc: Vector[NumTot]

        return ChemicalState(
            prim_aq_conc=new_conc[:len(chms.prim_aq_conc)], # type: ignore
            min_conc=chms.min_conc,
            sec_conc=new_conc[len(chms.prim_aq_conc):], # type: ignore
        )

@dataclass(frozen=True)
class RtStep(StepResult[NDArray]):
    """Holds the results of a single time step for a ReactiveTransportZone."""
    state: NDArray
    forc_flux: NDArray
    vap_flux: NDArray
    lat_flux: NDArray
    vert_flux: NDArray


class ReactiveTransportZone(Zone[NDArray, RtForcing, RtStep]):
    """A zone that solves reactive transport using a generic, function-based approach.

    This class acts as an ODE solver and orchestrator. It is initialized with
    functions that define the specific transport and kinetic reaction logic.
    This allows for a flexible definition of the model's biogeochemistry without
    hard-coding it into the zone itself.

    The `step` method uses operator splitting:
    1. Solves the ODE for transport and kinetic reactions.
    2. Solves for instantaneous chemical equilibrium.
    """

    _reaction_fn: Callable[[NDArray, RtForcing, dict[str, Any]], NDArray]
    _transport_fn: Callable[[NDArray, RtForcing, NDArray, dict[str, Any]], NDArray]
    _equilibrium_fn: Callable[[NDArray, RtForcing, dict[str, Any]], NDArray]
    params: dict[str, Any]
    name: str

    def __init__(
        self,
        reaction_fn: Callable[[NDArray, RtForcing, dict[str, Any]], NDArray],
        transport_fn: Callable[[NDArray, RtForcing, NDArray, dict[str, Any]], NDArray],
        equilibrium_fn: Callable[[NDArray, RtForcing, dict[str, Any]], NDArray],
        params: dict[str, Any],
        name: str = "unnamed",
    ):
        """Initializes the zone with specific logic functions and parameters.

        Args:
            reaction_fn: A function calculating dC/dt from reactions.
            transport_fn: A function calculating dC/dt from transport.
            equilibrium_fn: A function that solves for equilibrium concentrations.
            params: A dictionary of parameters for the provided functions.
            name: The name of this zone type.
        """
        self._reaction_fn = reaction_fn
        self._transport_fn = transport_fn
        self._equilibrium_fn = equilibrium_fn
        self.params = params
        self.name = name

    def _mass_balance_ode(self, c: NDArray, d: RtForcing, q_in: NDArray) -> NDArray:
        """Calculates the net rate of change of concentration (dC/dt).

        This method is conceptually based on the equation from the documentation:
        dm/dt = (dm/dt)_reaction + (dm/dt)_transport
        It calls the external functions provided during initialization.
        """
        reaction_rate = self._reaction_fn(c, d, self.params)
        transport_rate = self._transport_fn(c, d, q_in, self.params)
        return reaction_rate + transport_rate

    def step(self, s_0: NDArray, d: RtForcing, dt: float, q_in: NDArray) -> RtStep:
        """Advances the chemical state by one time step."""

        # 1. Solve the ODE for kinetic reactions and transport
        def f(t: float, c: NDArray) -> NDArray:
            return self._mass_balance_ode(c, d, q_in)

        res = solve_ivp(f, (0, dt), y0=s_0, dense_output=True)
        c_after_kinetics = res.y[:, -1]

        # 2. Solve for instantaneous chemical equilibrium
        c_new = self._equilibrium_fn(c_after_kinetics, d, self.params)

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

    # def solve_equilibrium(self, c: ChemicalState) -> ChemicalState:
    #     """
    #     Solve for the equilibrium concentrations
    #     """