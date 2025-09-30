from __future__ import annotations
from typing import TypeAlias, Callable, Any
from numpy.typing import NDArray
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .interfaces import Zone, StepResult


ChemicalState: TypeAlias = NDArray
"""Type alias for the chemical state of a zone, typically an array of concentrations."""


@dataclass(frozen=True)
class RtForcing:
    """Contains the hydrologic drivers for a reactive transport step for a single zone.

    Attributes:
        temp: Temperature (°C) for rate calculations.
        s: Hydrologic state (storage) of the zone (mm).
        q_in: Total incoming water flux from connected zones (mm/day).
        q_lat_out: Outgoing lateral water flux from this zone (mm/day).
        q_vert_out: Outgoing vertical water flux from this zone (mm/day).
    """

    temp: float
    s: float
    q_in: float
    q_lat_out: float
    q_vert_out: float


@dataclass(frozen=True)
class RtStep(StepResult[NDArray]):
    """Holds the results of a single time step for a ReactiveTransportZone.

    This is an immutable data structure that contains the new state and the
    calculated fluxes for a single zone over a single time step (`dt`). The state
    and fluxes are NumPy arrays to account for multiple chemical species.

    Attributes:
        state: The updated chemical state (e.g., concentrations) of the zone.
        forc_flux: The flux of chemicals into the zone from external forcing.
        vap_flux: The flux of chemicals out of the zone (e.g., volatilization).
        lat_flux: The lateral flux of chemicals out of the zone.
        vert_flux: The vertical flux of chemicals out of the zone.
    """

    state: NDArray
    forc_flux: NDArray
    vap_flux: NDArray
    lat_flux: NDArray
    vert_flux: NDArray


class ReactiveTransportZone(Zone[NDArray, RtForcing, RtStep]):
    """A concrete class for a single-zone reactive transport model.

    This class uses a composition-based approach, where the specific reaction
    and transport logic are provided as functions during initialization. This
    allows for flexible model definition without requiring subclassing.
    """

    _reaction_fn: Callable[[NDArray, RtForcing, dict[str, Any]], NDArray]
    _transport_fn: Callable[[NDArray, RtForcing, NDArray, dict[str, Any]], NDArray]
    params: dict[str, Any]
    name: str

    def __init__(
        self,
        reaction_fn: Callable[[NDArray, RtForcing, dict[str, Any]], NDArray],
        transport_fn: Callable[[NDArray, RtForcing, NDArray, dict[str, Any]], NDArray],
        params: dict[str, Any],
        name: str = "unnamed",
    ):
        """Initializes the ReactiveTransportZone with specific logic and parameters.

        Args:
            reaction_fn: A function that calculates the rate of change
                due to biogeochemical reactions.
            transport_fn: A function that calculates the rate of change
                due to advective-dispersive transport.
            params: A dictionary of parameters required by the reaction and
                transport functions.
            name: The name of this zone type.
        """
        self._reaction_fn = reaction_fn
        self._transport_fn = transport_fn
        self.params = params
        self.name = name

    def _mass_balance(self, c: NDArray, d: RtForcing, q_in: NDArray) -> NDArray:
        """Calculates the net rate of change of concentration (dC/dt)."""
        reaction = self._reaction_fn(c, d, self.params)
        transport = self._transport_fn(c, d, q_in, self.params)
        return reaction + transport

    def step(self, s_0: NDArray, d: RtForcing, dt: float, q_in: NDArray) -> RtStep:
        """Advances the chemical state by one time step.

        Args:
            s_0: The initial chemical state (concentrations) of the zone.
            d: The hydrologic forcing data for the current time step.
            dt: The duration of the time step in days.
            q_in: The total incoming chemical mass flux from other connected zones.

        Returns:
            An `RtStep` object containing the new state and all calculated fluxes.
        """

        def f(t: float, c: NDArray) -> NDArray:
            return self._mass_balance(c, d, q_in)

        res = solve_ivp(f, (0, dt), y0=s_0, dense_output=True)
        c_new = res.y[:, -1]

        # For this simple model, assume no direct chemical forcing or vaporization
        forc_flux = np.zeros_like(c_new)
        vap_flux = np.zeros_like(c_new)

        # Partition the outgoing flux into lateral and vertical components
        # based on the proportions of the water fluxes.
        total_q_out_water = d.q_lat_out + d.q_vert_out
        if total_q_out_water > 1e-9:
            # Outgoing mass flux is based on the average concentration over the step
            c_avg = res.sol(dt / 2)
            total_mass_out_flux = total_q_out_water * c_avg
            lat_flux = total_mass_out_flux * (d.q_lat_out / total_q_out_water)
            vert_flux = total_mass_out_flux * (d.q_vert_out / total_q_out_water)
        else:
            lat_flux = np.zeros_like(c_new)
            vert_flux = np.zeros_like(c_new)

        return RtStep(
            state=c_new,
            forc_flux=forc_flux,
            vap_flux=vap_flux,
            lat_flux=lat_flux,
            vert_flux=vert_flux,
        )

    def param_list(self) -> list[float]:
        """Returns the list of parameter values for the zone."""
        return [v for v in self.params.values() if isinstance(v, (float, int))]

    def columns(self, zone_id: int) -> list[str]:
        """Gets the column names for this zone for the output DataFrame."""
        # Assuming one chemical species for now
        name = f"{self.name}_{zone_id}"
        return [
            f"c_{name}",
            f"j_forc_{name}",
            f"j_vap_{name}",
            f"j_lat_{name}",
            f"j_vert_{name}",
        ]
